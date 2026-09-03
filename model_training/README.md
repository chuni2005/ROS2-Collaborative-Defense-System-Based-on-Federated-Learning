# model_training/ 腳本總覽

本檔案列出 `run_*.py` 與資料/評估工具腳本，依「對報告的重要性」分三類：

- **主線交付**：報告要引用的頭條數字，直接由這支腳本產出
- **支撐對照**：baseline、對照組、敏感度檢查——用來佐證或排除主線數字的替代解釋，本身不是頭條數字
- **決策紀錄**：調參或方法論驗證，決定了主線實驗該用什麼設定，不直接進報告

不含 `server.py`、`client.py`、`settings.py`、`main.py`——那些是系統本體，不是單次執行的實驗腳本。

---

## 主線交付

### `run_attack_experiment.py`（任務 13）

- **用途**：標籤翻轉攻擊驗收關卡。跑兩組完整 10 輪聯邦 bagging（`no_attack` / `attack_c1_full`，client 1 標籤翻轉 rate=1.0），無任何防禦，回答「攻擊是否造成可測量的傷害」。
- **對應 notes**：`notes/13-attack-injection.md`
- **產出結果檔**：
  - `results/attack_injection_summary.csv` — 兩組 × 10 輪的逐輪 accuracy/F1/margin（chunk_6）
  - `results/attack_injection_<label>_recall.txt` — 兩組第 10 輪模型的逐攻擊類型 recall（chunk_6）
  - `model_attack_no_attack/`、`model_attack_attack_c1_full/` — 兩組的 round-10 模型檔，被 `run_err_lfr_experiment.py` 直接重用為「無防禦」兩格，不重新訓練
- **報告數字**：`notes/13-attack-injection.md` 結論——chunk_6 上整體 F1 0.9720→0.9612（下降 1.08 個百分點）、`nmap discovery` recall 0.8194→0.7321（下降 8.73 個百分點）；`metasploit SYN flood` recall 兩組相同（0.9993），無影響。
  同一組模型在 test_natural/test_rare 上重新評估的數字（`nmap discovery` recall 0.8341→0.7373）改由 `run_err_lfr_experiment.py` 產出，見下方——兩邊數字不同是因為評估資料集不同（chunk_6 vs. test_natural/test_rare），不是矛盾。

### `run_err_lfr_experiment.py`（任務 14）

- **用途**：ERR/LFR 四格實驗矩陣 {無防禦, ERR/LFR} × {無攻擊, 有攻擊}。無防禦兩格沿用 `run_attack_experiment.py` 留下的 round-10 模型，ERR/LFR 兩格（`--aggregation=err_lfr`）由本檔案新跑，最終報告數字統一在 test_natural/test_rare 上重新評估。
- **對應 notes**：`notes/14-err-lfr-experiment.md`
- **產出結果檔**：
  - `results/err_lfr_summary.csv` — 四格 × 10 輪逐輪 accuracy/F1（chunk_6）
  - `results/err_lfr_participation.csv` — 四格 × 10 輪參與節點數
  - `results/err_lfr_exclusions.csv` — ERR/LFR 兩格逐輪排除了哪些 client_id
  - `results/err_lfr_final_report.csv` — 四格在 test_natural 上的整體指標
  - `results/err_lfr_final_recall_<cell>.txt` — 四格在 test_rare 上的逐攻擊類型 recall
- **報告數字**：`notes/14-err-lfr-experiment.md` 結論——
  1. 有攻擊+有防禦幾乎完全回復：`attack_err_lfr` test_natural F1=0.9719 vs. `no_attack_no_defense` F1=0.9726（差 0.07pp）；`nmap discovery` recall 從 0.7373 拉回 0.8261，補回缺口的 91.7%。
  2. 無攻擊+有防禦的誤刪率不接近 0（10 輪全部排除至少 1 個誠實節點），但最終 F1（0.9727）跟無防禦 baseline（0.9726）幾乎相同，誤判未造成可測量傷害。
  3. 存活節點數：`no_attack_err_lfr` 每輪 3–4/5 擺盪，`attack_err_lfr` 穩定 4/5（惡意節點十輪全被排除）。

---

## 支撐對照

### `run_winner_baseline.py`

- **用途**：在 `SPLIT_SEED=42` 重新切分後的資料上，重跑一次「贏者全拿」（`--aggregation=winner`，即 `our-project/` 現況）baseline，讓 bagging／ERR-LFR 的數字有對照基準。
- **對應 notes**：`notes/12-baseline.md`「2026-08-28 reseed 後對照」一節、`notes/12b-branch-delta.md`
- **產出結果檔**：`results/baseline_reseed_recall.txt`（第 10 輪模型逐攻擊類型 recall）；逐輪 accuracy/F1 只印在終端機，未落檔。
- **報告數字**：round-10 accuracy=0.9645、precision=0.9993、recall=0.9445、F1=0.9711（`notes/12-baseline.md` 第 62 行一帶），是後續所有 bagging／ERR-LFR 數字的比較基準線。

### `run_centralized_comparison.py`

- **用途**：把 5 個節點的訓練資料集中訓練成一個模型，當「架構問題 vs. 資料問題」的診斷對照組——某攻擊類型聯邦測不到時，用來確認是架構限制還是資料裡本來就沒訊號。不是要取代聯邦架構的方案。
- **對應 notes**：`notes/12-baseline.md`「三種偵測失敗類型的成因」一節
- **產出結果檔**：`model_centralized_reseed.ubj`、`results/centralized_reseed_recall.txt`
- **報告數字**：`ros2 reconnaissance` recall 上限（集中式）：新切分 90.37% vs. 舊切分 89.59%，核心結論不受影響——用來佐證聯邦架構下該類型測不到是架構限制，不是樣本量不足。

### `eval_baseline_on_test_natural.py`

- **用途**：一次性核對，不是可重跑管線。載入贏者全拿與 bagging 兩個既有 round-10 模型，在 test_natural.csv 上重算整體 F1 與逐攻擊類型 recall，跟原本只在 chunk_6 上量到的數字對照，同時驗證 `server.py` 新增的 `test_natural_data_path` 參數（`_evaluate_model_on_server(dataset="test_natural")`）接得起來。
- **對應 notes**：`notes/12c-val-test-split.md` 第 224–225 行
- **產出結果檔**：無。只印在終端機，不寫檔。
- **報告數字**：無獨立頭條數字——是任務 14 正式改用 test_natural/test_rare 評估流程之前的接線驗證，本身不被報告引用。

### `run_err_lfr_flip03.py`（任務 14 附加：攻擊強度敏感度）

- **用途**：把 `label_flip_rate` 從任務 14 的 1.0 降到 0.3，其餘設定完全比照任務 14，檢查 ERR/LFR 的偵測效果是否有下限。無防禦四格中的三格沿用既有結果，只新跑 `attack_flip0.3_err_lfr` 這一格。
- **對應 notes**：`notes/14-err-lfr-experiment.md` 第九節
- **產出結果檔**：
  - `results/err_lfr_flip03_summary.csv`、`_participation.csv`、`_exclusions.csv` — 逐輪合併結果／參與節點／排除紀錄
  - `results/err_lfr_flip03_candidate_a.csv` — 逐輪候選模型 A 的 accuracy/logloss
  - `results/err_lfr_flip03_client_impact.csv` — 逐輪每個節點的 err_impact/lfr_impact
  - `results/err_lfr_flip03_final_report.csv`、`err_lfr_final_recall_attack_flip0.3_err_lfr.txt`
- **報告數字**：`notes/14-err-lfr-experiment.md` §9.1——flip_rate 0.3 與 1.0 抓取效果一致，本次實驗沒有找到偵測下限；§9.3 舉出 round 2 的實測案例：惡意節點 client 1 的 `lfr_impact`（0.009851）比誠實節點（-0.0067~-0.0076）高出約三個數量級，ERR／LFR 聯集規則在這一輪確實發揮作用（代價是誤刪一個誠實節點）。

### `split.py`

- **用途**：把原始資料水塘抽樣切成 `split_data/chunk_1..6.csv`（5 個節點訓練資料 + 1 份伺服器驗證資料），是所有實驗腳本共用的資料切分基礎設施。固定隨機種子 `SPLIT_SEED=42`。
- **對應 notes**：`notes/12b-branch-delta.md` 第 12 項（補上固定種子的決策紀錄）、`notes/11-dataset-check.md`
- **產出結果檔**：`split_data/chunk_1.csv` ~ `chunk_6.csv`（不在 `results/` 底下，是所有實驗的輸入而非產出）
- **報告數字**：無獨立數字，但所有下游報告數字都依附在這份切分上——換種子或重新切分會讓既有數字失去可比性（`notes/12b-branch-delta.md` 已記錄過一次這種失效）。

### `build_test_sets.py`

- **用途**：從 `split.py` 抽樣後的剩餘資料池切出 `test_natural.csv`（依原始類型比例，對照整體 F1）與 `test_rare.csv`（稀有攻擊類型全數納入，用於逐類型 recall），兩者互不重疊。固定隨機種子 `SEED=42`（與 `split.py` 的 `SPLIT_SEED` 是不同次抽樣）。
- **對應 notes**：`notes/12c-val-test-split.md`
- **產出結果檔**：`split_data/test_natural.csv`、`split_data/test_rare.csv`
- **報告數字**：無獨立數字，但任務 14 起所有「報告用」的整體 F1／逐攻擊類型 recall 數字都讀這兩份檔案，不再讀 chunk_6（`notes/12c-val-test-split.md`「三份資料的用途規範」表）。

### `analyze_recall_by_attack.py`

- **用途**：對一個已訓練模型算逐攻擊類型 recall（observe 算 FPR），被 `run_leaf_scale_sweep.py`、`run_winner_baseline.py`、`run_attack_experiment.py` 等腳本呼叫，是所有「逐攻擊類型 recall」數字的共用計算邏輯，本身不是獨立實驗。
- **對應 notes**：`notes/13a-bagging-baseline.md`、`notes/13a-leaf-scale-fix.md`
- **產出結果檔**：無獨立輸出，印到 stdout，由呼叫端導向各自的 `*_recall.txt`。
- **報告數字**：無獨立數字——是其他腳本報告數字背後的計算方式。

---

## 決策紀錄

### `run_leaf_scale_sweep.py`

- **用途**：掃描 bagging 合併用的縮放係數 `leaf_scale` ∈ {1/5, 1/3, 1/2, 1}，解決「5 節點修正直接加總導致預測值暴衝」的問題，決定後續所有 bagging／ERR-LFR 實驗該用哪個值。
- **對應 notes**：`notes/13a-bagging-baseline.md`（問題發現）、`notes/13a-leaf-scale-fix.md`（掃描結果與定案）
- **產出結果檔**：`results/leaf_scale_summary.csv`（四組 × 10 輪逐輪 accuracy/F1/margin）、`results/leaf_scale_<label>_recall.txt`（四組第 10 輪逐攻擊類型 recall）
- **決策結果**：`leaf_scale=1/2`（0.5）定案——1/5、1/3、1/2 三者都穩定收斂（F1 0.9708/0.9711/0.9715），且 F1 與 `ros2 reconnaissance` recall 隨係數增加單調變好，1/2 是測試範圍內最好的；`leaf_scale=1`（不縮放）重現 margin 暴衝。此後所有 bagging／ERR-LFR 實驗（`run_loo_impact.py`、`run_attack_experiment.py`、`run_err_lfr_experiment.py`）都固定用 `leaf_scale=0.5`。

### `run_loo_impact.py`

- **用途**：留一法（LOO）訊號底噪重新量測——比較「全部節點都在」的模型 A 與「拿掉節點 i」的模型 B_i，量出拿掉一個誠實節點造成多大的 F1 變化，用來判斷這套量測方法本身雜訊有多大，是否夠格拿來當 ERR/LFR 判斷節點可疑的依據。是量測，不是真正剔除節點的防禦機制本身。
- **對應 notes**：`notes/13a-loo-impact-fixed.md`
- **產出結果檔**：`results/loo_impact_summary.csv`（六組 × 10 輪逐輪 accuracy/F1/margin）；impact 表只印在終端機。
- **決策結果**：修正合併 bug、套用 `leaf_scale=0.5` 後重新量測，impact 全距 0.0005（0.05 個百分點），五個節點全部由負轉正（拿掉任何一個誠實節點都讓 F1 下降），但全距仍跟「client 間本地表現差異」（0.06 個百分點）同一量級——訊號存在但很弱，是任務 14 解讀 ERR/LFR 結果時「誤判但沒造成傷害」這個現象的前置依據。

---

## 附註

三支任務 14/13a 相關的舊版留一法量測（沿用 Flower 官方 `aggregate()`，直接合併 round-10 完整模型）已知有 bug（只取到每個模型第一棵樹），對應舊數字記在 `notes/12-baseline.md`，**不可再引用**；`run_loo_impact.py` 是修正後的重新量測，兩次量測的差異本身也是有意義的發現，細節見 `notes/13a-loo-impact-fixed.md`。
