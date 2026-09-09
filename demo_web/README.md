# demo_web

機台監控儀表板。Flask（後端）+ 不用 build 工具的 Vue 3（前端），單一 process、單一 port，沒有 npm/Vite/CORS 這些額外的東西，故意做得越輕量越好。

```
demo_web/
├── backend/
│   ├── app.py                  # 主要 Flask app
│   ├── fdo_client.py           # 背景輪詢 FDO Owner server，快取「哪些機台真的上線了」
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

{"score": 87.5}
```

1. **關卡 A — FDO 身分**：`X-Device-Guid` 要能在 `guid_machine_map.json` 查到對應機台，且該機台要被 FDO Owner server 承認「已完成上線」。任一項不過 → `403`。這一關的資料來源是 `fdo-integration/`（見該資料夾的 README），不是這裡的程式碼自己生成的。
2. **關卡 B — 信任分數**：`score` 低於 `SCORE_THRESHOLD`（預設 50）持續超過 `ABNORMAL_SUSTAIN_SECONDS`（預設 4 秒）→ 觸發截斷 `BLOCK_DURATION_SECONDS`（預設 7 秒）。截斷期間收到的請求一樣回 `200 {"status": "dropped"}`，但資料不會被處理。這樣可以在 demo 時清楚示範「信任分數不只是顯示，還真的會拿來擋可疑機台」。

兩關分開設計、分開回應（403 vs 200 dropped），這樣可以清楚展示「陌生裝置直接被拒」跟「合法裝置但分數持續異常被截斷」是兩種不同的狀況。

`compute_diagnosis()`（`app.py` 裡）目前是一個固定回傳高分的 stub，是留給接 AI 判斷邏輯的人（負責 API 串接的組員）替換的——如果 POST body 裡有帶 `score` 就直接用那個值，沒帶才會落到這個 stub。

## 前端

Vue 3 Composition API，但整個專案沒有 build 步驟：瀏覽器原生 `<script type="module">` + vendor 進來的 `vue.esm-browser.prod.js`，元件用 template 字串寫在 `app.js` 裡。`api.js` 的 `usePolling` 是共用的輪詢 composable，各面板（狀態燈、版1、版2）都是每秒打一次對應的 API。

## requirements.txt

只有 `flask`。FDO Owner server 的輪詢刻意用標準庫 `urllib.request`，沒有另外加 `requests` 依賴。
