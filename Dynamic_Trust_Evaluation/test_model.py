import re
import json
import math
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

class Logger(object):
    def __init__(self, filename="evaluation.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class FuzzyDynamicTrustEngine:
    def __init__(self, config_path='fuzzy_threshold_config.json'):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            self.config = {re.sub(r"[\[\]<>]", "_", str(k)): v for k, v in raw_config.items()}
        except Exception as e:
            print(f"⚠️ [警告] 載入模糊設定檔失敗: {e}")
            self.config = {}

    def _calculate_triangle_membership(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x < c:
            return (c - x) / (c - b) if c != b else 1.0
        return 0.0

    def fuzzify_metric(self, config_key, value):
        conf = self.config[config_key]
        L_verts = conf["L_vertices"]
        M_verts = conf["M_vertices"]
        H_verts = conf["H_vertices"]

        mu_L = 1.0 if value <= L_verts[1] else self._calculate_triangle_membership(value, *L_verts)
        mu_M = self._calculate_triangle_membership(value, *M_verts)
        mu_H = 1.0 if value >= H_verts[1] else self._calculate_triangle_membership(value, *H_verts)
        return mu_L, mu_M, mu_H

    def defuzzify_risk_level(self, env_context):
        mu_L_list, mu_M_list, mu_H_list = [], [], []

        for feat, val in env_context.items():
            if feat in self.config:
                if math.isnan(val) or val == -1:
                    continue
                
                l, m, h = self.fuzzify_metric(feat, val)
                mu_L_list.append(l)
                mu_M_list.append(m)
                mu_H_list.append(h)

        if not mu_L_list:
            return 0.0

        rule_safe = sum(mu_L_list) / len(mu_L_list)
        rule_warn = max(mu_M_list)
        rule_danger = max(mu_H_list)

        numerator = (rule_safe * -1.0) + (rule_warn * 0.0) + (rule_danger * 1.0)
        denominator = rule_safe + rule_warn + rule_danger

        return 0.0 if denominator == 0 else numerator / denominator

    def evaluate_trust(self, trust_score, env_context, base_threshold=0.50, max_adj=0.10):
        env_risk_score = self.defuzzify_risk_level(env_context)
        dynamic_threshold = base_threshold + (env_risk_score * max_adj)
        decision = "ALLOW" if trust_score >= dynamic_threshold else "DENY"
        return decision, dynamic_threshold

def convert_type(value):
    if (isinstance(value, (int, float, np.number)) and not pd.isna(value)) and not isinstance(value, bool):
        return value
    if pd.isna(value) or value == "":
        return ""
    try:
        return pd.to_numeric(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != "attack"]

    if hasattr(df, "map"):
        df[feature_cols] = df[feature_cols].map(convert_type)
    else:
        df[feature_cols] = df[feature_cols].applymap(convert_type)

    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    if "attack" in df.columns:
        attack_mapping = {
            "observe": 0,
            "metasploit SYN flood": 1,
            "nmap discovery": 1,
            "nmap SYN flood": 1,
            "ros2 node crashing": 1,
            "ros2 reconnaissance": 1,
            "ros2 reflection": 1,
        }
        df["attack"] = df["attack"].replace(attack_mapping).infer_objects(copy=False)
        df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)

    df = df.drop(
        columns=[col for col in df.columns if "Unnamed" in col or "timestamp" in col],
        errors="ignore",
    )

    non_numeric_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns
    for col in non_numeric_cols:
        if col != "attack":
            df[col] = pd.Categorical(df[col]).codes

    df.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in df.columns]
    return df.astype(np.float32)

def evaluate_model(model_path: str | Path, test_data) -> None:
    model_path = Path(model_path)
    
    # 初始化模型
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    expected_cols = booster.feature_names
    
    current_dir = Path(__file__).resolve().parent
    json_path = current_dir / 'fuzzy_threshold_config.json'
    engine = FuzzyDynamicTrustEngine(str(json_path))

    all_y_true = []
    all_y_pred_dynamic = []

    if isinstance(test_data, pd.DataFrame):
        chunks = [test_data.copy()]
        test_source_name = "DataFrame"
    else:
        chunks = pd.read_csv(Path(test_data), low_memory=False, chunksize=100000)
        test_source_name = str(Path(test_data))

    for chunk_df in chunks:
        target_col = next((c for c in ['attack', 'label', 'Label', 'class', 'Attack'] if c in chunk_df.columns), None)
        if target_col:
            original_attack_labels = chunk_df[target_col].values
        else:
            original_attack_labels = ["Unknown"] * len(chunk_df)

        df = preprocess_data(chunk_df)

        if "attack" not in df.columns:
            raise ValueError("The test file does not contain an 'attack' column.")

        X = df.drop(columns=["attack"])
        y_true_chunk = df["attack"].astype(int).values

        if expected_cols:
            missing_cols = set(expected_cols) - set(X.columns)
            if missing_cols:
                missing_df = pd.DataFrame(-1, index=X.index, columns=list(missing_cols), dtype=np.float32)
                X = pd.concat([X, missing_df], axis=1)
            
            X = X[expected_cols]
        
        X = X.astype(np.float32)
        dtest = xgb.DMatrix(X)
        
        # 分數軟化魔法
        raw_margins = booster.predict(dtest, output_margin=True)
        T = 3.0 
        y_prob = 1.0 / (1.0 + np.exp(-raw_margins / T))

        X_dicts = X.to_dict(orient='records')

        for i in range(len(y_prob)):
            trust_score = 1.0 - float(y_prob[i])
            env_context = X_dicts[i]
            
            decision, dyn_th = engine.evaluate_trust(trust_score=trust_score, env_context=env_context)
            is_allowed = bool(decision == "ALLOW")
            
            all_y_pred_dynamic.append(0 if is_allowed else 1)
            all_y_true.append(y_true_chunk[i])

            output_data = {
                "封包類別": str(original_attack_labels[i]),
                "門檻": round(float(dyn_th), 4),
                "分數": round(float(trust_score), 4),
                "是否通過": is_allowed
            }
            
            print(json.dumps(output_data, ensure_ascii=False))
            sys.stdout.flush()
            
            time.sleep(0.01)

    y_pred = np.array(all_y_pred_dynamic)
    y_true = np.array(all_y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf = confusion_matrix(y_true, y_pred)

    print("\n=== Global Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Test data: {test_source_name}")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


# if __name__ == "__main__":
#     base_dir = Path(__file__).resolve().parent
    
#     log_path = base_dir / "evaluation.log"
#     sys.stdout = Logger(filename=str(log_path))
    
#     model_path = base_dir / "global_model_latest_0908.ubj"
#     test_csv_path = base_dir / "test-001.csv"
    
#     if not test_csv_path.exists():
#         print(f"❌ 找不到測試資料，請確認 {test_csv_path} 是否存在！")
#         sys.exit(1)

#     evaluate_model(model_path, test_csv_path)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    
    # 你的模型路徑
    model_path = base_dir / "global_model_latest_0908.ubj"
    
    # 剛剛分割出來的資料夾路徑
    split_dir = base_dir / "attack_dataset_splits_001"
    
    if not split_dir.exists():
        print(f"❌ 找不到分割資料夾，請確認 {split_dir} 是否存在！")
        sys.exit(1)

    # 找出資料夾內所有的 CSV 檔案
    csv_files = list(split_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ {split_dir} 裡面沒有任何 CSV 檔案！")
        sys.exit(1)

    original_stdout = sys.__stdout__

    for csv_file in csv_files:
        log_path = base_dir / f"evaluation_{csv_file.stem}.log"
        
        sys.stdout = Logger(filename=str(log_path))
        
        print(f"\n" + "="*50)
        print(f"🚀 開始評估資料集: {csv_file.name}")
        print(f"📁 儲存日誌於: {log_path.name}")
        print(f"="*50 + "\n")
        
        evaluate_model(model_path, csv_file)
        
        if hasattr(sys.stdout, 'log'):
            sys.stdout.log.close()
            
        sys.stdout = original_stdout
        
    print("\n✅ 所有 7 個分割檔案皆已評估完畢，Log 已全部分別儲存！")
