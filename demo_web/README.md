# demo_web

機台監控儀表板。Flask（後端）+ 不用 build 工具的 Vue 3（前端），單一 process、單一 port，沒有 npm/Vite/CORS 這些額外的東西，故意做得越輕量越好。

```
demo_web/
├── backend/
│   ├── app.py                  # 主要 Flask app
│   ├── fdo_client.py           # 背景輪詢 FDO Owner server，快取「哪些機台真的上線了」
│   ├── dynamic_trust.py        # 動態信任門檻（Dynamic_Trust_Evaluation 的 Fuzzy 引擎精簡版）
│   ├── guid_machine_map.json   # (執行期產生，gitignore) FDO guid -> 機台編號 對照表
│   └── requirements.txt
└── frontend/
    ├── index.html / style.css
    ├── api.js                  # 呼叫後端 API 的小函式 + usePolling 輪詢 composable
    ├── app.js                  # 所有 Vue 元件
    └── vendor/vue.esm-browser.prod.js   # vendor 進來的 Vue 3 production build，不連 CDN
```

## 畫面配置

照組員的手繪草圖：

- 左上：機台（1~5）下拉選單 + 確認鈕
- 右上：版1 / 版2 各自獨立的機台選擇器（決定中間、右下兩個面板各自要看哪一台）
- 左下：5 台機台的簡略狀態燈（顏色 + 分數 + FDO 上線徽章）
- 中間（版1）：AI 信任分數診斷——狀態文字、詳細資訊、信任分數
- 右下（版2）：機台 publisher 運作狀態——目前是否被截斷、截斷紀錄

## 怎麼跑

```bash
pip install -r backend/requirements.txt
python backend/app.py
```

開 `http://localhost:5181/`。在 `fdo-integration/` 那邊還沒幫任何機台完成上線之前，所有機台的 FDO 徽章會是 ✗，`/api/ingest` 一律回 403——這是預期行為。要接上真的機台身分驗證，請看 `../fdo-integration/README.md`。

## 後端 API

| Method | 路徑 | 用途 |
|---|---|---|
| GET | `/` | 回傳前端首頁（Flask 直接 serve 靜態檔案） |
| GET | `/api/machines` | 機台清單（id、name） |
| GET | `/api/machines/status` | 全部機台的簡略狀態：燈號、分數、`fdoOnboarded`、`fdoStale` |
| POST | `/api/machines/select` | 選定「目前作用機台」（左上角那組選單用） |
| POST | `/api/ingest` | 機台把資料丟進來的入口，見下方「`/api/ingest` 的兩道關卡」 |
| GET | `/api/machines/<id>/diagnosis` | 版1：該機台的信任分數診斷 |
| GET | `/api/machines/<id>/publisher-status` | 版2：該機台目前是否被截斷、截斷紀錄 |

## `/api/ingest` 的兩道關卡

呼叫方式：

```
POST /api/ingest
X-Device-Guid: <該機台上線後拿到的 FDO guid>
Content-Type: application/json

{"score": 87.5, "threshold": 62.3, "passed": true}
```

`threshold`、`passed` 都是選填的。

1. **關卡 A — FDO 身分**：`X-Device-Guid` 要能在 `guid_machine_map.json` 查到對應機台，且該機台要被 FDO Owner server 承認「已完成上線」。任一項不過 → `403`。這一關的資料來源是 `fdo-integration/`（見該資料夾的 README），不是這裡的程式碼自己生成的。
2. **關卡 B — 信任分數**：機台被判定「異常」持續超過 `ABNORMAL_SUSTAIN_SECONDS`（預設 4 秒）→ 觸發截斷 `BLOCK_DURATION_SECONDS`（預設 7 秒）。截斷期間收到的請求一樣回 `200 {"status": "dropped"}`，但資料不會被處理。這樣可以在 demo 時清楚示範「信任分數不只是顯示，還真的會拿來擋可疑機台」。「是否異常」怎麼判定，見下面「`passed` 由誰決定」。

兩關分開設計、分開回應（403 vs 200 dropped），這樣可以清楚展示「陌生裝置直接被拒」跟「合法裝置但分數持續異常被截斷」是兩種不同的狀況。

`compute_diagnosis()`（`app.py` 裡）目前是一個固定回傳高分的 stub，是留給接 AI 判斷邏輯的人（負責 API 串接的組員）替換的——如果 POST body 裡有帶 `score` 就直接用那個值，沒帶才會落到這個 stub。

### 信任分數門檻、是否通過：由評估器提供，或後端內部模擬

這組欄位設計對應到 `Dynamic_Trust_Evaluation/test_model.py` 裡 `evaluate_model()` 本來就會輸出的格式（`門檻`、`分數`、`是否通過`）——真正的評估流程（XGBoost 信任分數 + Fuzzy 動態門檻）算完之後，直接把 `threshold`、`score`、`passed` 一起傳給 `/api/ingest` 就好，demo_web 後端不需要自己知道那 35 個網路/系統特徵長什麼樣子，也不用重新判斷一次，只負責接收評估結果、執行「持續異常幾秒就截斷」這個狀態機。

**`threshold`（信任分數門檻）：**
- **有帶**（`threshold_source: "external"`）：直接採用呼叫端算好的值。
- **沒帶**（`threshold_source: "internal"`，例如 `simulate-ingest.sh` 只帶 `score` 的舊式呼叫）：後端用 `dynamic_trust.py`（`Dynamic_Trust_Evaluation/fuzzy_inference.py` 的精簡版，只保留「環境風險模糊推論」，不依賴 pandas/xgboost）自己模擬一個：

  ```
  threshold = (0.60 + env_risk_score × 0.10) × 100     # 結果落在 50~70 之間
  ```

  `env_risk_score` 目前是用 `fuzzy_threshold_config.json` 裡每個特徵本來就記錄的 `mu`/`sigma`（平均值/標準差）做常態分布隨機取樣模擬出來的，純粹是「還沒有真評估器時」的過渡 fallback。

**`passed`（是否通過，布林值）：**
- **有帶**：後端直接採信這個判斷（`abnormal = not passed`），**不會**再拿 `score` 跟 `threshold` 重新比較一次——即使 `score` 看起來明明高於 `threshold`，只要 `passed` 是 `false`，一樣視為異常。這樣評估器不用擔心後端的比較邏輯（例如 `<` 還是 `<=`）跟自己不一致。
- **沒帶**：後端才自己比較 `score < threshold`。

算出來的門檻、來源、（內部模擬時的）環境風險係數、是否異常存在每台機台的 `threshold`/`threshold_source`/`env_risk`/`abnormal` 欄位裡，`/api/machines/<id>/diagnosis` 會回傳 `threshold`，前端折線圖那條門檻虛線就是照這個值畫的。

## 前端

Vue 3 Composition API，但整個專案沒有 build 步驟：瀏覽器原生 `<script type="module">` + vendor 進來的 `vue.esm-browser.prod.js`，元件用 template 字串寫在 `app.js` 裡。`api.js` 的 `usePolling` 是共用的輪詢 composable，各面板（狀態燈、版1、版2）都是每秒打一次對應的 API。

## requirements.txt

只有 `flask`。FDO Owner server 的輪詢刻意用標準庫 `urllib.request`，沒有另外加 `requests` 依賴。
