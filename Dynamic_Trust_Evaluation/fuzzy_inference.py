import json
import glob
import pandas as pd
import random
import math  # 用來判斷 nan

class FuzzyDynamicTrustEngine:
    def __init__(self, config_path='fuzzy_threshold_config.json'):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"✅ 成功載入json檔，包含 {len(self.config)} 個特徵的模糊邊界設定。")
        except Exception as e:
            print(f"❌ 載入設定檔失敗: {e}")
            self.config = {}

    def _calculate_triangle_membership(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x < c:
            return (c - x) / (c - b) if c != b else 1.0
        return 0.0

    def fuzzify_metric(self, feature_name, value):
        if feature_name not in self.config:
            return 1.0, 0.0, 0.0

        conf = self.config[feature_name]
        L_verts = conf["L_vertices"]
        M_verts = conf["M_vertices"]
        H_verts = conf["H_vertices"]

        mu_L = 1.0 if value <= L_verts[1] else self._calculate_triangle_membership(value, *L_verts)
        mu_M = self._calculate_triangle_membership(value, *M_verts)
        mu_H = 1.0 if value >= H_verts[1] else self._calculate_triangle_membership(value, *H_verts)

        return mu_L, mu_M, mu_H

    def defuzzify_risk_level(self, env_context, verbose=False):
        mu_L_list = []
        mu_M_list = []
        mu_H_list = []

        if verbose:
            print("\n" + "-"*45)
            print(" 1. 動態模糊化 (Fuzzification) 細節")
            print("-"*45)

        for feat, val in env_context.items():
            if feat in self.config:
                # 防呆：遇到 NaN 略過
                if math.isnan(val):
                    continue
                
                l, m, h = self.fuzzify_metric(feat, val)
                mu_L_list.append(l)
                mu_M_list.append(m)
                mu_H_list.append(h)
                if verbose:
                    print(f"[{feat:^35}] 值={val:<12.4f} | 隸屬度: L={l:.3f}, M={m:.3f}, H={h:.3f}")

        if not mu_L_list:
            return 0.0

        # R1: 取所有有效特徵的平均安全度(代表環境的整體健康指數)
        # rule_safe = min(mu_L_list)
        rule_safe = sum(mu_L_list) / len(mu_L_list)
        
        # R2, R3: 依然取 MAX (只要任一特徵有異常/危險，立刻拉高警報)
        rule_warn = max(mu_M_list)
        rule_danger = max(mu_H_list)

        if verbose:
            print("\n" + "-"*45)
            print(" 2. 規則推論 (Rule Evaluation) 結果")
            print("-"*45)
            print(f" R1 (整體健康度)   -> 取 L 的平均值: SAFE = {rule_safe:.4f}")
            print(f" R2 (任一異常即警告) -> 取 M 的最大值: WARN = {rule_warn:.4f}")
            print(f" R3 (任一危險即阻擋) -> 取 H 的最大值: DANGER = {rule_danger:.4f}")

        # 重心法解模糊化
        numerator = (rule_safe * -1.0) + (rule_warn * 0.0) + (rule_danger * 1.0)
        denominator = rule_safe + rule_warn + rule_danger

        env_risk_score = 0.0 if denominator == 0 else numerator / denominator

        if verbose:
            print("\n" + "-"*45)
            print(" 3. 解模糊化 (Defuzzification)")
            print("-"*45)
            print(f" 最終環境風險係數 (env_risk_score): {env_risk_score:.4f}")

        return env_risk_score

    def evaluate_trust(self, trust_score, env_context, base_threshold=0.60, max_adj=0.10, verbose=False):
        env_risk_score = self.defuzzify_risk_level(env_context, verbose)
        dynamic_threshold = base_threshold + (env_risk_score * max_adj)
        decision = "ALLOW" if trust_score >= dynamic_threshold else "DENY"
        
        if verbose:
            print("\n" + "="*45)
            print(" 零信任動態門檻決策結果")
            print("="*45)
            print(f" XGBoost 傳入的基礎信任分數: {trust_score:.4f}")
            print(f" 動態門檻計算: {base_threshold} + ({env_risk_score:.4f} * {max_adj})")
            print(f" 調整後的實際門檻: {dynamic_threshold:.4f}")
            print(f" 最終系統決策: >> {decision} <<")
            
        return decision, dynamic_threshold, env_risk_score



# 資料集抽取封包測試 
def extract_real_packets(target_normal=2, target_attack=2):
    csv_files = glob.glob('ROSPaCe_reduced/rospace_reduced_*.csv')
    if not csv_files:
        print("❌ 找不到 CSV 檔案！請確認資料夾路徑。")
        return [], []

    header_file = 'ROSPaCe_reduced/rospace_reduced_0.csv'
    try:
        correct_columns = pd.read_csv(header_file, nrows=0).columns.tolist()
    except:
        print(f"❌ 讀取 {header_file} 表頭失敗。")
        return [], []

    normal_collected = []
    attack_collected = []
    
    # 隨機打亂檔案順序，確保每次測試的封包來源多樣化
    random.shuffle(csv_files)
    
    print(f"\n開始跨檔案搜尋，目標：{target_normal} 個正常封包, {target_attack} 個攻擊封包...")

    for file in csv_files:
        # 如果數量已湊齊，提早結束搜尋
        if len(normal_collected) >= target_normal and len(attack_collected) >= target_attack:
            break

        print(f"  📂 正在掃描 {file}...")
        try:
            # 為了避免記憶體爆炸，每次只讀取 10 萬筆
            if 'rospace_reduced_0.csv' in file:
                df = pd.read_csv(file, nrows=100000, low_memory=False)
                row_offset = 2 # 0號檔案有表頭，所以 index 0 = CSV 裡的第 2 行
            else:
                df = pd.read_csv(file, names=correct_columns, nrows=100000, low_memory=False, header=None)
                row_offset = 1 # 1~14號沒表頭，所以 index 0 = CSV 裡的第 1 行
            
            label_col = next((c for c in ['label', 'Label', 'class', 'Attack', 'attack'] if c in df.columns), None)
            if not label_col:
                continue
                
            normal_df = df[df[label_col] == 'observe']
            attack_df = df[df[label_col] != 'observe']
            
            # 抽取缺少的正常封包
            need_normal = target_normal - len(normal_collected)
            if need_normal > 0 and not normal_df.empty:
                sampled = normal_df.sample(min(need_normal, len(normal_df)))
                for idx, row in sampled.iterrows():
                    pkt = row.to_dict()
                    pkt['_source_file'] = file
                    pkt['_source_row'] = idx + row_offset
                    normal_collected.append(pkt)

            # 抽取缺少的攻擊封包
            need_attack = target_attack - len(attack_collected)
            if need_attack > 0 and not attack_df.empty:
                sampled = attack_df.sample(min(need_attack, len(attack_df)))
                for idx, row in sampled.iterrows():
                    pkt = row.to_dict()
                    pkt['_source_file'] = file
                    pkt['_source_row'] = idx + row_offset
                    attack_collected.append(pkt)

        except Exception as e:
            print(f"  ⚠️ 讀取 {file} 失敗: {e}")
            continue

    return normal_collected, attack_collected

# 執行測試
if __name__ == "__main__":
    engine = FuzzyDynamicTrustEngine('fuzzy_threshold_config.json')
    
    # 在設定抽取的數量
    TARGET_NORMAL = 2
    TARGET_ATTACK = 2
    
    normal_pkts, attack_pkts = extract_real_packets(TARGET_NORMAL, TARGET_ATTACK)

    # ---------------- 測試正常封包 ----------------
    for i, pkt in enumerate(normal_pkts, 1):
        print("\n\n" + "🟢"*5 + f" 測試 [正常封包 {i}/{len(normal_pkts)}] " + "🟢"*5)
        print(f" 📄 來源檔案: {pkt['_source_file']}")
        print(f" 🔢 CSV 行號: 第 {pkt['_source_row']} 行 (可直接開啟原檔核對)")
        print(f" 🏷️ 真實標籤: observe")
        
        env_context = {}
        for k, v in pkt.items():
            if k in engine.config:
                try: env_context[k] = float(v)
                except: pass
                
        # 假設 XGBoost 判定為 0.95 (高度信任)
        engine.evaluate_trust(trust_score=0.95, env_context=env_context, verbose=True)

    # ---------------- 測試攻擊封包 ----------------
    for i, pkt in enumerate(attack_pkts, 1):
        lbl_key = next((k for k in ['label', 'Label', 'class', 'Attack', 'attack'] if k in pkt), 'Unknown')
        real_attack_name = pkt.get(lbl_key, 'Unknown')
        
        print("\n\n" + "🔴"*5 + f" 測試 [攻擊封包 {i}/{len(attack_pkts)}] " + "🔴"*5)
        print(f" 📄 來源檔案: {pkt['_source_file']}")
        print(f" 🔢 CSV 行號: 第 {pkt['_source_row']} 行 (可直接開啟原檔核對)")
        print(f" 🏷️ 真實標籤: {real_attack_name}")
        
        env_context = {}
        for k, v in pkt.items():
            if k in engine.config:
                try: env_context[k] = float(v)
                except: pass
                
        # 假設 XGBoost 判定為 0.15 (極度不信任)
        engine.evaluate_trust(trust_score=0.15, env_context=env_context, verbose=True)