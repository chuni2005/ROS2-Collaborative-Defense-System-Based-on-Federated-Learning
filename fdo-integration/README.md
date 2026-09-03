# FDO ↔ demo_web 整合

讓 `demo_web` 裡的機台身分是真的——機台要先真的跑過一輪 FIDO Device Onboard（FDO）上線流程，`/api/ingest` 才會接受它的資料，取代原本寫死的 `IP_MACHINE_MAP`。

`/api/ingest` 有兩道獨立關卡，都要過（AND）：

- **關卡 A — FDO 身分**：呼叫端要帶 `X-Device-Guid: <guid>`。後端會查這個 guid 有沒有對應到機台編號（`demo_web/backend/guid_machine_map.json`），以及 Owner server 現在是不是承認它「已完成上線」（`to2_completed: true`）。任一項不過 → `403`。
- **關卡 B — 信任分數**（既有邏輯，沒有改動）：如果機台正處於分數觸發的截斷窗口內，request 會被接受但安靜 drop 掉（`200 {"status": "dropped"}`）。

不需要本機裝 Go——`go-fdo-server` 跟 `go-fdo-client` 都在 Docker 裡跑，從 `../FDO/` 底下釘住的 submodule commit 建置。

## 一次性設定 + 每台機台上線

在這個目錄（`fdo-integration/`）底下，用 Git Bash 執行：

```bash
./scripts/01-gen-certs.sh          # 只需跑一次
./scripts/02-up-servers.sh         # 只需跑一次 —— 啟動三個 container，等 /health
./scripts/03-configure-rvinfo.sh   # 只需跑一次 —— 設定 RVInfo + device-CA 信任 + RVTO2Addr

# 針對機台 1~5，各跑一次：
./scripts/04-onboard-machine.sh 1
./scripts/04-onboard-machine.sh 2
./scripts/04-onboard-machine.sh 3
./scripts/04-onboard-machine.sh 4
./scripts/04-onboard-machine.sh 5
```

每支 script 都會透過 `lib.sh` 自動設定 `MSYS_NO_PATHCONV=1`，不用自己在指令前面加——這是為了讓 Git Bash 不要把傳給 `docker run`/`docker compose run` 的容器內路徑（例如 `/certs/...`、`/workdir/...`）誤轉成 Windows 路徑。如果某個指令還是怪怪的，改用 PowerShell 跑同一支 script 就好——底層呼叫的 `docker` CLI 指令是一樣的。

每次跑 `04-onboard-machine.sh` 都會在 `../demo_web/backend/guid_machine_map.json` 裡新增/更新一筆紀錄。針對同一台機台重複執行是安全的（模擬重新上線——該機台對應的 guid 會被換掉）。

## 啟動儀表板

```bash
pip install -r ../demo_web/backend/requirements.txt   # 第一次才需要
python ../demo_web/backend/app.py
```

會在 `:5181` 啟動 Flask，同時開一條背景執行緒每隔幾秒去問 Owner server 目前的裝置清單。打開 `http://localhost:5181/`——第一輪輪詢跑完後，已上線的機台會在狀態清單顯示 FDO 徽章。

## Demo 流量（排練用）

```bash
./scripts/simulate-ingest.sh 1 95   # 已上線機台、分數正常 -> 200 ok
./scripts/simulate-ingest.sh 1 20   # 連續丟 4 秒以上 -> 觸發截斷/drop 窗口
curl -X POST http://localhost:5181/api/ingest -d '{"score":95}'  # 沒帶/錯誤的 GUID -> 403
```

## 給負責寫真正機台模擬器的人

每次 `POST /api/ingest` 都要帶 `X-Device-Guid: <guid>`，`<guid>` 就是那台機台在 `guid_machine_map.json` 裡的紀錄（跑完 `04-onboard-machine.sh` 之後去讀）。JSON body 的形狀不變（`{"score": ...}`）。

## 關閉

```bash
./scripts/05-teardown.sh            # 停掉 container，保留憑證/憑證資料/資料庫
./scripts/05-teardown.sh --purge    # 連同 workdir/ 跟 guid_machine_map.json 一起清掉
```

## 已知取捨（是刻意的決定，不是漏洞）

- **裝置初始化（DI）目前沒有身分驗證**：這個版本的程式碼，任何能連到 Manufacturing container 的人都可以完成 DI、拿到 voucher。本機 demo 沒差——這也是為什麼「FDO 本身只解決身分，不解決授權」。
- **Owner server 的 `/api/v1/*` 管理 API 目前也沒有掛 auth**（這個 checkout 的 middleware chain 裡沒接 AuthN/AuthZ）。一樣只限本機 demo 使用。
- **FDO 狀態過期時採 fail-open**：如果 Owner server 或 `fdo_client.py` 的輪詢超過 `FDO_STALE_AFTER_SECONDS`（預設 30 秒）沒有成功更新，`/api/ingest` 會繼續用最後已知的快取狀態，而不是直接把所有請求都擋掉，dashboard 上會顯示 `fdoStale` 標記。短暫過期的信任訊號，總比 demo 現場整個 dashboard 變黑好。
