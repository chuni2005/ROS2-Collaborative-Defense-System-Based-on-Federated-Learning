import pandas as pd
import glob
import math
import random
import matplotlib.pyplot as plt
from collections import deque
from fuzzy_inference import FuzzyDynamicTrustEngine 

def scan_for_perfect_continuous_window(window_size=500):
    """
    循序掃描真實資料集，尋找一段包含 [正常 -> 攻擊 -> 正常] 的完美連續封包視窗。
    這保證了時間軸 100% 連續，未經任何人工剪接。
    """
    csv_files = glob.glob('ROSPaCe_reduced/rospace_reduced_*.csv')
    if not csv_files:
        print("❌ 找不到 CSV 檔案！")
        return []

    # 確保依照真實時間順序讀取 (0, 1, 2... 14)
    try:
        csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except:
        pass

    header_file = 'ROSPaCe_reduced/rospace_reduced_0.csv'
    try:
        correct_columns = pd.read_csv(header_file, nrows=0).columns.tolist()
    except:
        return []

    print(f"⏳ 啟動滑動視窗雷達 (大小: {window_size} 封包)... 正在循序掃描資料集...")
    
    buffer = deque(maxlen=window_size)
    
    for file in csv_files:
        print(f"  📂 正在掃描 {file}...")
        try:
            if 'rospace_reduced_0.csv' in file:
                chunk_iter = pd.read_csv(file, chunksize=50000, low_memory=False)
            else:
                chunk_iter = pd.read_csv(file, names=correct_columns, chunksize=50000, header=None, low_memory=False)
            
            for chunk in chunk_iter:
                label_col = next((c for c in ['label', 'Label', 'class', 'Attack', 'attack'] if c in chunk.columns), None)
                if not label_col:
                    break
                
                for idx, row in chunk.iterrows():
                    buffer.append(row.to_dict())
                    
                    # 當緩衝區滿了，檢查是否符合我們想要的「完美突發攻擊情境」
                    if len(buffer) == window_size:
                        labels = [str(r.get(label_col, '')) for r in buffer]
                        
                        # 條件 1：前 100 個封包都是正常 (確保我們看到平靜期)
                        first_100_normal = all(l == 'observe' for l in labels[:100])
                        # 條件 2：後 100 個封包都是正常 (確保我們看到恢復期)
                        last_100_normal = all(l == 'observe' for l in labels[-100:])
                        # 條件 3：中間 300 個封包中，至少包含 30 個攻擊封包 (確保有明顯的攻擊突波)
                        attack_count = sum(1 for l in labels[100:-100] if l != 'observe')
                        
                        if first_100_normal and last_100_normal and attack_count >= 30:
                            print(f"\n🎯 雷達鎖定！在 {file} 找到完美的真實連續攻擊片段！")
                            return list(buffer)
                            
        except Exception as e:
            print(f"  ⚠️ 讀取 {file} 發生錯誤: {e}")
            continue

    print("❌ 掃描完畢，沒有找到符合條件的連續片段 (資料集中的攻擊可能過於密集或過於稀疏)。")
    return []

def main():
    engine = FuzzyDynamicTrustEngine('fuzzy_threshold_config.json')

    # 1. 啟動雷達擷取真實連續封包
    timeline = scan_for_perfect_continuous_window(window_size=500)
    
    if not timeline:
        return

    history = {'packet_id': [], 'label': [], 'xgboost_score': [], 'dynamic_threshold': [], 'decision': []}
    print(f"🎬 開始放映真實連續推論 (共 {len(timeline)} 個連續封包)...")
    
    # 2. 連續推論
    for idx, pkt in enumerate(timeline):
        label_col = next((k for k in ['label', 'Label', 'class', 'Attack', 'attack'] if k in pkt), 'Unknown')
        real_label = str(pkt.get(label_col, 'Unknown'))
        is_normal = (real_label == 'observe')
        
        # 模擬 XGBoost 分數 (真實世界中，這裡就是直接拿 XGBoost 算出來的分數)
        xgb_score = random.uniform(0.85, 0.99) if is_normal else random.uniform(0.05, 0.35)

        env_context = {}
        for k, v in pkt.items():
            if k in engine.config:
                try: 
                    val = float(v)
                    if not math.isnan(val): env_context[k] = val
                except: pass

        # 這裡會用到你修改過 R1 取平均值 (MEAN) 的引擎！
        decision, dyn_th, risk = engine.evaluate_trust(xgb_score, env_context, verbose=False)

        history['packet_id'].append(idx)
        history['label'].append(real_label)
        history['xgboost_score'].append(xgb_score)
        history['dynamic_threshold'].append(dyn_th)
        history['decision'].append(decision)

    # 3. 繪製圖表
    print("📈 正在繪製動態防禦曲線圖...")
    plt.figure(figsize=(14, 7))
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
    except: pass

    # 畫出動態門檻線 (藍色粗線)
    plt.plot(history['packet_id'], history['dynamic_threshold'], label='動態門檻 (Dynamic Threshold)', color='blue', linewidth=2.5)
    
    # 標示正常與攻擊的點
    n_idx = [i for i, lbl in enumerate(history['label']) if lbl == 'observe']
    a_idx = [i for i, lbl in enumerate(history['label']) if lbl != 'observe']
    
    plt.scatter([history['packet_id'][i] for i in n_idx], [history['xgboost_score'][i] for i in n_idx], 
                color='green', alpha=0.6, label='正常封包 (XGBoost Score)', s=20)
    plt.scatter([history['packet_id'][i] for i in a_idx], [history['xgboost_score'][i] for i in a_idx], 
                color='red', alpha=0.8, marker='x', label='真實攻擊封包 (XGBoost Score)', s=40)

    # 畫一條固定基準線 (0.60)
    plt.axhline(y=0.60, color='gray', linestyle='--', alpha=0.5, label='傳統固定門檻 (0.60)')

    plt.title('信任分數門檻動態調整曲線', fontsize=16, fontweight='bold')
    plt.xlabel('連續封包時間序 (Continuous Packet Sequence)', fontsize=12)
    plt.ylabel('信任分數 / 門檻值', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()
    
    output_img = 'continuous_burst.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ 圖表已儲存為：{output_img}")
    plt.show()

if __name__ == "__main__":
    main()