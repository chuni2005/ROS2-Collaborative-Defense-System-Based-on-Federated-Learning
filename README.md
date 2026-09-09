# ROS2 Collaborative Defense System (Federated Learning)

以聯邦學習（Federated Learning）為基礎的 ROS2 網路入侵偵測/協同防禦系統專題。整個 repo 分成三個主要部分：

- `model_training/` — 聯邦學習模型訓練（偵測機台是否被攻擊、給出信任分數）
- `demo_web/` — 監控儀表板（展示各機台狀態、AI 診斷分數、有沒有被截斷）
- `fdo-integration/`（搭配 `FDO/`）— 用 FIDO Device Onboard 讓機台身分是真的上線驗證過的，不是隨便一個來源就能冒充

三者的關係：機台先透過 FDO 完成上線拿到身分 → 機台把資料送進系統，AI 模型（`model_training/` 訓出來的）判斷信任分數 → 分數送進 `demo_web/` 的後端，分數持續異常會被截斷，並即時顯示在儀表板上。

---

## `model_training/` — 聯邦學習訓練

用 [Flower](https://flower.ai/)（`flwr`）做聯邦學習框架，模型是 XGBoost，資料集是 ROSPaCe 系列的 ROS2 網路封包/系統指標資料（`dataset/`，欄位包含封包資訊、系統資源指標、ROS2 pub/sub 資訊，以及訓練用的 `attack` 標籤）。

**重要檔案**

| 檔案 | 用途 |
|---|---|
| `main.py` | 進入點，跑 `MainRunner` 的主迴圈：切資料 → 開 server → 開 clients → 監控存活 → 換下一輪資料 |
| `settings.py` | 所有設定（`NUM_CLIENTS`、`NUM_ROUNDS`、資料路徑、切割大小等）都在這裡的常數，`MainRunner` 也定義在這 |
| `split.py` | 把大資料集切成 N 份（`NUM_CLIENTS` 份給各個 client，加 1 份給 server 驗證用） |
| `client.py` | Flower client：讀自己那份資料、前處理（`attack` 欄位轉成 0/1 標籤）、訓練 XGBoost、跟 server 交換參數 |
| `server.py` | Flower server：聚合各 client 的模型參數，跑指定輪數 |

**怎麼跑**

```bash
pip install flwr xgboost pandas numpy scikit-learn
cd model_training
python main.py
```

資料集要放在 `settings.py` 裡 `TARGET_DATA` 指定的路徑（預設對應到 repo 根目錄的 `dataset/`，會依 `NUM_CLIENTS+1` 切成好幾份丟到 `split_data/`）。

**注意**

- `dataset/*` 已經在 `.gitignore` 排除（檔案是 GB 等級的，不要進版控）
- `attack` 欄位只能拿來訓練，不能當作 demo/驗證時的信任依據

---

## `demo_web/` — 監控儀表板

Flask（後端）+ 不用 build 工具的 Vue 3（前端），單一 process、單一 port，越輕量越好，沒有 npm/Vite。

```
demo_web/
├── backend/
│   ├── app.py                  # 主要 Flask app：機台狀態、ingest、診斷、publisher 狀態等 API
│   ├── fdo_client.py           # 背景輪詢 FDO Owner server，快取「哪些機台真的上線了」
│   ├── guid_machine_map.json   # (執行期產生，gitignore) FDO guid -> 機台編號 對照表
│   └── requirements.txt
└── frontend/
    ├── index.html / style.css
    ├── api.js                  # 呼叫後端 API 的小函式 + usePolling 輪詢
    ├── app.js                  # 所有 Vue 元件（機台選擇、狀態燈、版1診斷、版2 publisher狀態）
    └── vendor/vue.esm-browser.prod.js
```

**畫面配置**：左上機台選擇+確認鈕、右上版1/版2各自要看哪台機台的下拉選單、左下 5 台機台簡略狀態燈（含分數、FDO 上線徽章）、中間版1（AI 信任分數診斷）、右下版2（機台 publisher 運作狀態/截斷紀錄）。

**核心邏輯：`/api/ingest` 兩道關卡**

1. **關卡 A（FDO 身分）**：request 要帶 `X-Device-Guid` header，後端查這個 guid 是否對應到機台、且 FDO Owner server 確認它「已完成上線」。任一項不過 → `403`。
2. **關卡 B（信任分數）**：分數（`score` 欄位，0~100）低於 `SCORE_THRESHOLD`（預設 50）持續超過 `ABNORMAL_SUSTAIN_SECONDS`（預設 4 秒）→ 觸發截斷 `BLOCK_DURATION_SECONDS`（預設 7 秒），期間收到的資料一樣回 200 但直接 drop 掉，不處理。

這樣可以清楚示範兩種情境：陌生/沒上線過的裝置直接被拒絕，跟「合法上線過的機台，但信任分數持續異常」被安靜截斷，是兩種不同的故障模式。

**怎麼跑**

```bash
pip install -r demo_web/backend/requirements.txt
python demo_web/backend/app.py
```

開 `http://localhost:5181/`。在 FDO 還沒接好之前，所有機台的 FDO 徽章會顯示 ✗（尚未上線），`/api/ingest` 一律回 403——這是預期行為，要先跑過 `fdo-integration/` 的上線流程才會變成 ✓。

---

## `fdo-integration/` + `FDO/` — 機台身分上線（FIDO Device Onboard）

`FDO/` 是三個 vendor 進來的 git submodule（`go-fdo` 協定函式庫、`go-fdo-client` 裝置端上線工具、`go-fdo-server` 可以跑 manufacturing/rendezvous/owner 三種角色的伺服器），`fdo-integration/` 是我們自己寫的一批 orchestration script，把這三個 submodule 兜起來給 `demo_web/` 用，**不會去動 submodule 裡的任何檔案**。

**為什麼要有這塊**：原本機台身分是後端寫死一張「來源 IP → 機台編號」的表，沒有安全性可言。現在改成機台要先真的跑過一輪 FDO 協定（跟 Manufacturing server 做裝置初始化、跟 Owner/Rendezvous server 換憑證完成上線），系統才會發給它一個裝置 GUID，`demo_web` 的後端會拿這個 GUID 去問 Owner server「這台真的上線了嗎」，過了才處理它的資料。

```
fdo-integration/
├── docker-compose.client.yaml   # 從本地 FDO/go-fdo-client submodule 建置 client image
├── scripts/
│   ├── lib.sh                   # 共用函式（健康檢查、RVInfo/device-CA/RVTO2Addr 設定、voucher 交換等）
│   ├── 01-gen-certs.sh          # 產生 manufacturer/device_ca/owner 金鑰憑證
│   ├── 02-up-servers.sh         # 啟動 manufacturer/rendezvous/owner 三個 container
│   ├── 03-configure-rvinfo.sh   # 一次性伺服器設定（上線前一定要跑）
│   ├── 04-onboard-machine.sh    # 單一機台的完整上線流程
│   ├── 05-teardown.sh           # 關閉 container（--purge 可連同憑證/資料庫一起清掉）
│   └── simulate-ingest.sh       # 排練用：讀某機台的 guid，直接送測試分數給 demo_web
├── workdir/                     # (執行期產生，gitignore) 憑證、sqlite db、各機台的上線憑證
└── README.md                    # 完整跑法細節
```

**完整跑法**（Git Bash，在 `fdo-integration/` 目錄下）：

```bash
./scripts/01-gen-certs.sh
MSYS_NO_PATHCONV=1 ./scripts/02-up-servers.sh
./scripts/03-configure-rvinfo.sh

# 針對機台 1~5 各跑一次：
MSYS_NO_PATHCONV=1 ./scripts/04-onboard-machine.sh 1
# ...重複到 5

# 回到 demo_web 啟動後端
python ../demo_web/backend/app.py
```

跑完之後，`demo_web/backend/guid_machine_map.json` 會有每台機台對應的 guid，儀表板上該機台的 FDO 徽章會變成 ✓。詳細的每一步說明、坑點（例如一定要跑 `03-configure-rvinfo.sh`，不然 TO0/TO1/TO2 會失敗但看不出原因）都寫在 `fdo-integration/README.md` 裡。

**需要的工具**：Docker（跑 `go-fdo-server`/`go-fdo-client`，不需要本機裝 Go）、`openssl`。

**已知取捨（不是 bug，是刻意的決定）**：
- 裝置初始化（DI）目前沒有白名單機制，本機 demo 沒差
- Owner server 的管理 API 目前沒有掛 auth，純本機 demo 沒問題
- 如果 Owner server 或輪詢暫時斷線，後端採「fail-open + 標示過期」而不是直接把整個 dashboard 擋死
