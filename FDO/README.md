# FDO/

這個資料夾底下是三個 vendor 進來的 [FIDO Device Onboard](https://fidoalliance.org/specs/FDO/FIDO-Device-Onboard-PS-v1.1-20220419/FIDO-Device-Onboard-PS-v1.1-20220419.html)（FDO）官方 Go 實作，各自都是獨立的 **git submodule**（有自己的 `.git`、獨立版本歷史），不是複製貼上進來的程式碼：

| 資料夾 | 上游專案 | 用途 |
|---|---|---|
| `go-fdo/` | [fido-device-onboard/go-fdo](https://github.com/fido-device-onboard/go-fdo) | FDO 協定本體的 Go 函式庫 |
| `go-fdo-client/` | [fido-device-onboard/go-fdo-client](https://github.com/fido-device-onboard/go-fdo-client) | 裝置端（機台）用來跑上線流程的 CLI 工具 |
| `go-fdo-server/` | [fido-device-onboard/go-fdo-server](https://github.com/fido-device-onboard/go-fdo-server) | 伺服器端，可以跑 Manufacturing / Rendezvous / Owner 三種角色 |

## 這是什麼、為什麼會在這個專案裡

FDO 解決的是「一台裝置在沒有預先設定密鑰的情況下，怎麼證明自己是誰」——裝置先跟 Manufacturing server 做初始化（DI）、換到一張擁有權憑證（Ownership Voucher），再跟 Rendezvous/Owner server 走一輪協定（TO0 → TO1 → TO2）完成上線，最後拿到一個穩定的裝置 GUID。

這個 repo 把 FDO 拿來取代 `demo_web/` 後端原本寫死的「來源 IP → 機台編號」對照表：機台要先真的跑過上線流程、Owner server 承認它「已完成上線」，`demo_web` 的 `/api/ingest` 才會接受它送來的資料。

## 這裡不是拿來直接用的

- **不要修改這三個資料夾裡的任何檔案**——它們是 submodule，改了會產生難以 review 的 submodule diff。要調整怎麼啟動/設定，去改 `../fdo-integration/` 底下我們自己寫的 script，不要動這裡。
- **不要隨便 `git submodule update --remote`**——commit 已經釘住，`../fdo-integration/` 裡的腳本是照著目前這個特定 commit 的行為寫的（例如 `go-fdo-client print` 沒有結構化輸出、要靠正則撈 GUID），亂更新版本可能會讓那些腳本失效。

## 實際怎麼被使用

三個 submodule 本身只是原始碼，真正「怎麼建置、怎麼啟動、怎麼串起來給機台上線」的邏輯都在 `../fdo-integration/`（一批 shell script + docker compose 設定）。要跑起來、或想知道細節，請看 `../fdo-integration/README.md`。

## Clone 這個 repo 時要注意

Submodule 預設不會自動抓下來，`git clone` 完之後要多跑一次：

```bash
git submodule update --init --recursive
```

或是 clone 的時候就直接帶 `--recurse-submodules`。
