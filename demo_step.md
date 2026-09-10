# Demo 步驟:用真正訓練好的模型把分數推進網頁

這份文件從零開始，一路做到「真正訓練好的 XGBoost 模型自己算分數、自己判斷通過或不通過，即時推進網頁」這個完整流程，每一步都附上實際執行過的成功 log，照著打就能在別台電腦上重現。

全部指令都要在 **Git Bash** 裡打，不是 cmd，不是 PowerShell。

如果不知道怎麼開 Git Bash:在專案資料夾裡，滑鼠右鍵點一下空白處，選單裡如果有「Git Bash Here」就點它。

想單純測網頁後端的判斷邏輯(不需要模型，用手動 curl 打分數)，請看 [test_step.md](test_step.md)。

---

## 第 -1 步:確認你的 FDO 資料夾底下三個資料夾不是空的

```bash
git submodule update --init --recursive
```

---

## 第 0 步:確認你在對的資料夾

打開 Git Bash 之後，先打:

```bash
pwd
```

應該要印出類似這樣的東西(結尾是專案名稱):
```
/.../ROS2-Collaborative-Defense-System-Based-on-Federated-Learning
```

如果不是，用 `cd` 切過去

打完再打一次 `pwd` 確認位置對了，再打:

```bash
ls
```

應該要看到 `FDO`、`fdo-integration`、`demo_web` 這幾個資料夾名稱。看到才繼續下一步。

---

## 第 1 步:打開 Docker Desktop

打開「Docker Desktop」，等待開啟，回到 Git Bash，打:

```bash
docker version
```

**怎麼知道成功**:畫面印出兩大段文字，一段開頭是 `Client:`，一段開頭是 `Server:`，兩段都有印出來，中間沒有紅字錯誤訊息。

**實際執行畫面**(2026-09-10 本機實測):
```
Client:
 Version:           29.0.1
 API version:       1.52
 Go version:        go1.25.4
 Git commit:        eedd969
 Built:             Fri Nov 14 16:19:55 2025
 OS/Arch:           windows/amd64
 Context:           desktop-linux

Server: Docker Desktop 4.53.0 (211793)
 Engine:
  Version:          29.0.1
  API version:      1.52 (minimum version 1.44)
  Go version:       go1.25.4
  Git commit:       198b5e3
  Built:            Fri Nov 14 16:17:57 2025
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v2.1.5
  GitCommit:        fcd43222d6b07379a4be9786bda52438f0dd16a1
 runc:
  Version:          1.3.3
  GitCommit:        v1.3.3-0-gd842d771
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```
(版本號會依你安裝的 Docker Desktop 版本不同，重點是 Client/Server 兩段都印出來、沒有紅字錯誤。)

**如果失敗**(例如出現 `error during connect` 這種字):代表 Docker Desktop 還沒完全啟動好，再等一下，重新打一次 `docker version`。

---

## 第 2 步:切換到 fdo-integration 資料夾

```bash
cd fdo-integration
```

打 `pwd` 確認，結尾應該是 `.../fdo-integration`。

---

## 第 3 步:產生憑證

這一步是產生一些加密用的檔案，**它會自動用 Docker 跑一個小工具**，你不用自己額外下任何 docker 指令，一行下面的指令就搞定:

```bash
bash scripts/01-gen-certs.sh
```

**怎麼知道成功**:最後一行印出 `[fdo] certs ready in ...`，然後下面列出 6 個檔案(3 個 `.key`、3 個 `.crt`)。

**這一步只需要做一次**。如果你之後重跑一次這個指令，它會直接印 `certs already exist...skipping`，這是正常的，不是錯誤。

---

## 第 4 步:啟動三個 FDO 伺服器

```bash
bash scripts/02-up-servers.sh
```

這個指令內部會自動幫你「建置(build)Docker image」再「啟動(run)成 container」，這兩件事都包在這一支腳本裡面。

第一次跑這個指令，因為要下載/建置 image，**可能要花 1~3 分鐘**，請耐心等，不要中途按 Ctrl+C 中斷。

**怎麼知道成功**:最後印出 `[fdo] all three FDO server roles are up`。

想再進一步確認，可以打:

```bash
docker ps
```

應該要看到 3 行，名稱分別是 `manufacturer`、`rendezvous`、`owner`，狀態欄(STATUS)開頭是 `Up`(後面可能還會寫 `unhealthy` 或 `health: starting`，這個沒關係，它跟服務有沒有正常運作是兩回事)。

**實際執行畫面**(2026-09-10 本機實測):
```
CONTAINER ID   IMAGE              COMMAND                   CREATED        STATUS                            PORTS                                         NAMES
45e12e3d5548   go-fdo-server      "go-fdo-server --db-…"   8 days ago     Up 9 seconds (health: starting)   0.0.0.0:8041->8041/tcp, [::]:8041->8041/tcp   rendezvous
9beaf4d20e61   go-fdo-server      "go-fdo-server --db-…"   8 days ago     Up 9 seconds (health: starting)   0.0.0.0:8038->8038/tcp, [::]:8038->8038/tcp   manufacturer
4421d43e880c   go-fdo-server      "go-fdo-server --db-…"   8 days ago     Up 9 seconds (health: starting)   0.0.0.0:8043->8043/tcp, [::]:8043->8043/tcp   owner
```
(`CREATED` 顯示 8 天前是因為這幾個 container 是之前建的、這次只是重新啟動；你第一次跑會顯示剛建立。)

**這一步也只需要做一次**，除非你之後有跑過關閉的指令(第 8 步)。

---

## 第 5 步:設定伺服器之間的信任關係

```bash
bash scripts/03-configure-rvinfo.sh
```

**怎麼知道成功**:最後一行印出 `[fdo] server configuration complete`。

過程中會看到幾行 `curl: (22) The requested URL returned error: 404`，**這是正常的**，不是錯誤，不用理它。

**這一步也只需要做一次**。

---

## 第 6 步:讓機台上線

機台編號是 1~5 的數字。假設你要讓「機台 1」上線，打:

```bash
bash scripts/04-onboard-machine.sh 1
```

**怎麼知道成功**:過程中應該會看到這幾個關鍵字依序出現:
- `running Device Initialization (DI)`
- `DI GUID = ...`
- `FIDO Device Onboard Complete`
- 最後一行:`[fdo] machine 1 onboarded， guid=...`

看到最後這行，就代表機台 1 成功上線了。後面的步驟都用機台 1 示範，其他機台編號同理。

```bash
bash scripts/04-onboard-machine.sh 2
bash scripts/04-onboard-machine.sh 3
bash scripts/04-onboard-machine.sh 4
bash scripts/04-onboard-machine.sh 5
```

---

## 第 7 步:啟動網頁

先回到專案最上層資料夾:

```bash
cd ..
```

打 `pwd` 確認，結尾應該**不是** `fdo-integration`(是回到上一層了)。

安裝網頁需要的套件:

```bash
pip install -r demo_web/backend/requirements.txt
```

啟動網頁:

```bash
python demo_web/backend/app.py
```

**怎麼知道成功**:畫面印出類似:
```
* Serving Flask app 'app'
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5181
Press CTRL+C to quit
```
(以上是 2026-09-10 本機實測的完整開機畫面，WARNING 那行是 Flask 內建開發伺服器的固定提醒，不是錯誤。)

**這個視窗接下來不要關掉、不要按 Ctrl+C**，關掉網頁就會停止運作。如果之後要做別的事，開一個新的 Git Bash 視窗，不要動這個。

打開瀏覽器(Chrome、Edge 都可以)，網址列輸入 `http://localhost:5181`，應該會看到「機台監控 Demo」畫面，機台 1 旁邊有綠色的 `FDO ✓`。

---

## 第 8 步:讓真正的模型把分數推進網頁

前面都是環境建置，這一步才是這份文件的重點:讓 `Dynamic_Trust_Evaluation` 裡訓練好的 XGBoost 模型自己算分數、自己判斷通過或不通過，直接推進網頁，不用手動輸入任何數字。

**先準備好兩個檔案**，放到專案最上層的 `base/` 資料夾底下，**檔名要完全一樣**：

- `global_model_latest.ubj`(訓練好的模型；跑完 `model_training/main.py` 之後會在 `model_training/model/` 底下)
- `test.csv`(要拿來測試的資料；跑完 `model_training/main.py` 之後會在 `model_training/test-data/` 底下，也可以自己準備一份格式一樣的)

**第 7 步開的那個網頁視窗留著不要動**，開一個新的 Git Bash 視窗：

```bash
cd "你的專案路徑/Dynamic_Trust_Evaluation"
python push_to_web.py --machine 1 --count 20 --interval 1
```

- `--machine 1`：要送去哪一台機台，這台必須先用 `04-onboard-machine.sh` 上線過，不然會直接報錯叫你先上線。
- `--count 20`：從 `test.csv` 拿前幾筆資料來跑就好，不用整份幾百萬筆都跑(會跑很久)。
- `--interval 1`：每筆之間停 1 秒，方便對照網頁畫面的變化；不加這個參數預設也是 1 秒。

**怎麼知道成功**：畫面會一行一行印出，例如：

```
[0] 封包類別=observe 分數=... 門檻=... 是否通過=... -> {'machineId': 1, 'status': 'ok'}
```

回到瀏覽器的網頁，機台 1 的分數會即時更新；如果連續幾筆都被判定「不通過」，大約 4 秒後會變成 `'status': 'dropped'`，網頁上機台 1 的燈號會變黃色/紅色，右下角面板會顯示「已截斷」，詳細資訊的門檻會標示「(由評估器提供)」。

**實際執行畫面**(2026-09-10 本機實測，用剛訓練好的模型跑機台 1 的前 5 筆資料):

```
$ python push_to_web.py --machine 1 --count 5 --interval 1
[push_to_web] 機台 1 的 GUID: 6058923b71906461c5ca050e9c7c0f0c
[0] 封包類別=observe 分數=0.02 門檻=51.36 是否通過=False -> {'machineId': 1, 'status': 'ok'}
[1] 封包類別=observe 分數=0.0 門檻=51.52 是否通過=False -> {'machineId': 1, 'status': 'ok'}
[2] 封包類別=observe 分數=0.0 門檻=51.64 是否通過=False -> {'machineId': 1, 'status': 'ok'}
[3] 封包類別=observe 分數=0.0 門檻=46.73 是否通過=False -> {'machineId': 1, 'status': 'dropped'}
[4] 封包類別=observe 分數=0.0 門檻=51.22 是否通過=False -> {'machineId': 1, 'status': 'dropped'}
```

同一時間網頁後端那個視窗(第 7 步開的)印出對應的請求紀錄：

```
127.0.0.1 - - [10/Sep/2026 03:03:12] "POST /api/ingest HTTP/1.1" 200 -
127.0.0.1 - - [10/Sep/2026 03:03:15] "POST /api/ingest HTTP/1.1" 200 -
127.0.0.1 - - [10/Sep/2026 03:03:18] "POST /api/ingest HTTP/1.1" 200 -
127.0.0.1 - - [10/Sep/2026 03:03:21] "POST /api/ingest HTTP/1.1" 200 -
127.0.0.1 - - [10/Sep/2026 03:03:24] "POST /api/ingest HTTP/1.1" 200 -
```

跑完之後查詢機台 1 的診斷資訊：

```bash
curl -s http://localhost:5181/api/machines/1/diagnosis
```

```json
{"blocked":false,"blockedSecondsRemaining":0.0,"details":["最後更新: 03:03:18","信任分數門檻: 51.6（由評估器提供）"],"machineId":1,"score":0.0,"status":"異常","threshold":51.64}
```

`"status":"異常"`、門檻標示「由評估器提供」，證明網頁後端是真的採信模型算出來的門檻和分數，不是自己內部模擬的——「模型 → 網頁」這條路是通的。

**已知限制**：目前模型的特徵前處理(把 MAC 位址、通訊協定這類文字欄位轉成數字的方式)還不夠穩定，同一個值在不同批次可能被轉成不同數字，所以現在跑出來的「是否通過」不一定準，可能會比預期看到更多「不通過」(上面的實測紀錄也能看到，5 筆全是良性封包卻大部分判定不通過)。這是後續要修的已知問題，不影響這一步驗證的重點——「模型算出來的結果能不能正確送進網頁」這件事本身是沒問題的。

---

## 第 9 步:全部做完之後怎麼關掉

先回到 fdo-integration 資料夾:

```bash
cd fdo-integration
```

只是先關掉、之後還想繼續用(下次不用重跑第 3、4、5 步):

```bash
bash scripts/05-teardown.sh
```

想整個乾淨重來(下次要從第 3 步重新開始):

```bash
bash scripts/05-teardown.sh --purge
```

網頁那個視窗(第 7 步開的)，直接在那個視窗按 `Ctrl + C` 就能停掉。
