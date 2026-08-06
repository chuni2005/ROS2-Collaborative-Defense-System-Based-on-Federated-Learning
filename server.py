import argparse
import os
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_dir: str, num_clients: int):
        self.num_clients = num_clients
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.latest_model_path = os.path.join(self.model_dir, "global_model_latest.model")

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
        )

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        """
        Flower 在整個訓練開始前會呼叫這個方法一次，決定第一輪要用什麼
        全域模型。預設（回傳 None）會觸發 Flower 去跟隨機一個 client
        要初始參數（也就是原本 log 裡看到的
        "Requesting initial parameters from one random client"）。

        這裡改成：如果 model_dir 底下已經有上一次訓練存下的
        global_model_latest.model，就直接讀取它的 bytes，包成
        Parameters 回傳，當作這一次啟動的起始模型 —— 等於接續上一次
        訓練的結果繼續往下練，而不是每次啟動都從零開始。

        若檔案不存在（例如第一次執行、或 model_dir 是全新的資料夾），
        則回傳 None，維持 Flower 原本「跟隨機 client 要初始參數」的行為。
        """
        if os.path.exists(self.latest_model_path):
            with open(self.latest_model_path, "rb") as f:
                model_bytes = f.read()

            if model_bytes:
                print(
                    f"[Info] Found existing model at {self.latest_model_path}, "
                    f"resuming training from it instead of starting from scratch."
                )
                return Parameters(tensors=[model_bytes], tensor_type="xgboost-ubj")

        print(
            f"[Info] No existing model found at {self.latest_model_path}, "
            f"starting from scratch (will request initial parameters from a client)."
        )
        return None

    def _extract_payload(self, parameters: Optional[Parameters]) -> Optional[bytes]:
        """
        方案 B 下，client 端（XGBoostClient，繼承 fl.client.Client）
        已經直接把 xgb.Booster.save_model() 產生的 raw bytes 放進
        Parameters.tensors，沒有再經過 NumPyClient 的 numpy 序列化包裝，
        所以這裡不需要再做任何 np.load / np.frombuffer 的解碼，
        直接取出來的 bytes 就是合法的 xgboost 模型檔內容。
        """
        if parameters is None or not getattr(parameters, "tensors", None):
            return None
        if not parameters.tensors:
            return None
        return bytes(parameters.tensors[0])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            # 有 client 端 fit 失敗，這種情況值得留下紀錄以便排查
            print(f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}")

        payloads = []

        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            metrics = fit_res.metrics or {}
            accuracy = float(metrics.get("accuracy", 0.0))
            load_failures = int(metrics.get("model_load_failures", 0))

            if load_failures > 0:
                print(
                    f"[Warning] Round {server_round} client {client_proxy.cid} reported "
                    f"{load_failures} model load failure(s) so far. That client may have "
                    f"trained from scratch this round instead of continuing federated training."
                )

            if payload:
                payloads.append((payload, accuracy))
            else:
                print(f"[Warning] Round {server_round} client {client_proxy.cid} returned an empty payload.")

        if not payloads:
            print(f"[Warning] Round {server_round} has no valid model payloads.")
            return None, {}

        # 只保留分數最高的 client 模型，作為下一輪的全域模型。
        best_payload, best_accuracy = max(payloads, key=lambda item: item[1])

        model_path = os.path.join(self.model_dir, f"global_model_round_{server_round}.model")

        with open(model_path, "wb") as f:
            f.write(best_payload)
        with open(self.latest_model_path, "wb") as f:
            f.write(best_payload)

        print(f"[Info] Round {server_round} kept the highest-scoring model (accuracy={best_accuracy:.4f}) "
              f"and saved it to {model_path}")

        # tensor_type 標記與 client 端一致，方便日後追蹤格式來源；
        # tensors 直接放 raw bytes，不經過任何 numpy 轉換。
        aggregated_parameters = Parameters(tensors=[best_payload], tensor_type="xgboost-ubj")
        return aggregated_parameters, {"accuracy": best_accuracy}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            print(f"Round {server_round} has no evaluate results to aggregate.")
            return None, {}

        if failures:
            print(f"[Warning] Round {server_round} had {len(failures)} evaluate failure(s): {failures}")

        # 計算每個 client 評估結果的加權準確率與加權 loss
        accuracies = [float(r.metrics.get("accuracy", 0.0)) * r.num_examples for _, r in results]
        losses = [float(r.loss) * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        total_examples = sum(examples)
        if total_examples == 0:
            return None, {}

        weighted_accuracy = sum(accuracies) / total_examples
        weighted_loss = sum(losses) / total_examples

        # 診斷用：把每個 client 個別的 accuracy 印出來，方便判斷
        # 「accuracy 恆為 1.0」是全域現象還是只發生在特定 client（通常代表
        # 該 client 本地測試集只有單一類別，或發生資料洩漏)。
        per_client_acc = {
            client_proxy.cid: float(r.metrics.get("accuracy", 0.0))
            for client_proxy, r in results
        }
        print(f"[Diagnostic] Round {server_round} per-client accuracy: {per_client_acc}")
        print(f"[Info] Round {server_round} global validation accuracy: {weighted_accuracy:.4f}, "
              f"loss: {weighted_loss:.4f}\n")

        return weighted_loss, {"accuracy": weighted_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Flower server with XGBoost strategy.")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save the trained model.")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds to train the model.")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients to train the model.")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080", help="Flower server address")
    args = parser.parse_args()

    strategy = XGBoostStrategy(
        model_dir=args.model_dir,
        num_clients=args.num_clients
    )

    print("[Info] Flower Server (XGBoost) is starting...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )