"""Push real model predictions to the demo_web backend.

Reads rows from test.csv, runs them through the same XGBoost model +
FuzzyDynamicTrustEngine as test_model.py, and POSTs each row's
score/threshold/passed to demo_web's /api/ingest — using the target
machine's onboarded FDO GUID — so the web page shows results computed by
the real model instead of the backend's internal simulation.

Usage:
    python push_to_web.py --machine 1
    python push_to_web.py --machine 1 --count 50 --interval 0.5
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_model as tm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "base"
DEFAULT_BACKEND_URL = "http://localhost:5181"
DEFAULT_GUID_MAP_PATH = BASE_DIR.parent / "demo_web" / "backend" / "guid_machine_map.json"


def load_guid(machine_id: int, guid_map_path: Path) -> str:
    if not guid_map_path.exists():
        raise SystemExit(
            f"找不到 {guid_map_path}\n"
            f"請先跑 fdo-integration/scripts/04-onboard-machine.sh {machine_id} 讓機台上線"
        )
    data = json.loads(guid_map_path.read_text(encoding="utf-8"))
    for guid, info in data.items():
        if info.get("machineId") == machine_id:
            return guid
    raise SystemExit(
        f"機台 {machine_id} 尚未上線\n"
        f"請先跑 fdo-integration/scripts/04-onboard-machine.sh {machine_id}"
    )


def push_row(backend_url: str, guid: str, score: float, threshold: float, passed: bool) -> dict:
    body = json.dumps({"score": score, "threshold": threshold, "passed": passed}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend_url}/api/ingest",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Device-Guid": guid},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="把模型算出的信任分數即時推送到 demo_web 後端")
    parser.add_argument("--machine", type=int, required=True, help="機台編號 (1~5)，必須先用 04-onboard-machine.sh 上線")
    parser.add_argument("--model", default="global_model_latest.ubj", help="模型檔名/路徑，相對於 base/ 資料夾")
    parser.add_argument("--test-data", default="test.csv", help="測試資料檔名/路徑，相對於 base/ 資料夾")
    parser.add_argument("--count", type=int, default=20, help="要推送幾筆資料 (預設 20)")
    parser.add_argument("--interval", type=float, default=1.0, help="每筆之間間隔秒數 (預設 1 秒，跟網頁的節奏對得上)")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--guid-map", default=str(DEFAULT_GUID_MAP_PATH))
    args = parser.parse_args()

    guid = load_guid(args.machine, Path(args.guid_map))

    print(f"[push_to_web] 機台 {args.machine} 的 GUID: {guid}")

    booster = xgb.Booster()
    booster.load_model(str(DATA_DIR / args.model))
    expected_cols = booster.feature_names

    engine = tm.FuzzyDynamicTrustEngine(str(BASE_DIR / "fuzzy_threshold_config.json"))

    df = pd.read_csv(DATA_DIR / args.test_data, nrows=args.count, low_memory=False)
    original_labels = df["attack"].values if "attack" in df.columns else ["Unknown"] * len(df)

    proc = tm.preprocess_data(df)
    X = proc.drop(columns=["attack"])
    if expected_cols:
        missing_cols = set(expected_cols) - set(X.columns)
        if missing_cols:
            missing_df = pd.DataFrame(-1, index=X.index, columns=list(missing_cols), dtype=np.float32)
            X = pd.concat([X, missing_df], axis=1)
        X = X[expected_cols]
    X = X.astype(np.float32)

    dtest = xgb.DMatrix(X)
    raw_margins = booster.predict(dtest, output_margin=True)
    T = 3.0
    y_prob = 1.0 / (1.0 + np.exp(-raw_margins / T))

    X_dicts = X.to_dict(orient="records")

    for i in range(len(y_prob)):
        trust_score = 1.0 - float(y_prob[i])
        env_context = X_dicts[i]

        decision, dyn_th = engine.evaluate_trust(trust_score=trust_score, env_context=env_context)
        is_allowed = bool(decision == "ALLOW")

        # demo_web 的 /api/ingest 是 0~100 分尺度，engine 算出來的是 0~1，要換算。
        score_100 = round(trust_score * 100, 2)
        threshold_100 = round(dyn_th * 100, 2)

        try:
            result = push_row(args.backend_url, guid, score_100, threshold_100, is_allowed)
        except urllib.error.HTTPError as e:
            print(f"[{i}] HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
            continue
        except urllib.error.URLError as e:
            raise SystemExit(f"連不到 {args.backend_url}，demo_web 後端有開嗎？({e})")

        print(
            f"[{i}] 封包類別={original_labels[i]} 分數={score_100} 門檻={threshold_100} "
            f"是否通過={is_allowed} -> {result}"
        )

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
