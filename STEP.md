# 複刻步驟(從零開始，一步一步照做)

全部指令都要在 **Git Bash** 裡打，不是 cmd，不是 PowerShell。

如果不知道怎麼開 Git Bash:在專案資料夾裡，滑鼠右鍵點一下空白處，選單裡如果有「Git Bash Here」就點它。

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

應該要看到 3 行，名稱分別是 `manufacturer`、`rendezvous`、`owner`，狀態欄(STATUS)開頭是 `Up`(後面可能還會寫 `unhealthy`，這個沒關係，它跟服務有沒有正常運作是兩回事)。

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

## 第 6 步:讓機台上線(這一步可以重複做，每次換一個機台編號)

機台編號是 1~5 的數字。假設你要讓「機台 1」上線，打:

```bash
bash scripts/04-onboard-machine.sh 1
```

**怎麼知道成功**:過程中應該會看到這幾個關鍵字依序出現:
- `running Device Initialization (DI)`
- `DI GUID = ...`
- `FIDO Device Onboard Complete`
- 最後一行:`[fdo] machine 1 onboarded， guid=...`

看到最後這行，就代表機台 1 成功上線了。

**如果想讓機台 2、3、4、5 也上線**，重複打這行指令，只是把數字換掉:

```bash
bash scripts/04-onboard-machine.sh 2
bash scripts/04-onboard-machine.sh 3
bash scripts/04-onboard-machine.sh 4
bash scripts/04-onboard-machine.sh 5
```

一次只上線一台，一台跑完(看到 `onboarded， guid=...`)再跑下一台。想上線幾台都可以，不用五台都跑。

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
* Running on http://127.0.0.1:5181
```

**這個視窗接下來不要關掉、不要按 Ctrl+C**，關掉網頁就會停止運作。如果之後要做別的事，開一個新的 Git Bash 視窗，不要動這個。

---

## 第 8 步:打開網頁看結果

打開瀏覽器(Chrome、Edge 都可以)，網址列輸入:

```
http://localhost:5181
```

**怎麼知道成功**:看到「機台監控 Demo」的畫面，左下角有機台 1~5 的清單。剛剛第 6 步有跑過上線的機台(例如機台 1)，旁邊會有綠色的 `FDO ✓`;沒跑過上線的機台會是灰色的 `FDO ✗`。

---

## 第 9 步(可選):測試三種情境

**開一個新的 Git Bash 視窗**(不要用第 7 步那個，那個要留著跑網頁)，切換到專案資料夾再進 fdo-integration:

```bash
cd "你的專案路徑/fdo-integration"
```

**情境 A:沒有上線過的裝置，應該被拒絕**

```bash
curl -X POST http://localhost:5181/api/ingest -d '{"score":95}'
```
會印出 `{"error":"unknown device guid"，...}`，這是正確的結果。

**情境 B:已上線機台(用機台 1 舉例)、分數正常**

```bash
bash scripts/simulate-ingest.sh 1 95
```
會印出 `{"machineId":1，"status":"ok"}`，這是正確的結果。回到瀏覽器頁面，機台 1 的分數應該會更新成 95.0。

**情境 C:已上線機台、分數持續偏低，會被暫時封鎖**

把下面這行**連續打 4~5 次**，每次中間停個 1 秒左右(用手動一直按 Enter 重複執行就可以):

```bash
bash scripts/simulate-ingest.sh 1 20
```
前幾次會回 `{"machineId":1，"status":"ok"}`，第 4 次左右開始會變成 `{"machineId":1，"status":"dropped"}`——代表系統偵測到分數持續異常，開始擋掉這台機台的資料。回到瀏覽器頁面，右下角面板會顯示「已截斷」。

---

## 第 10 步:全部做完之後怎麼關掉

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

---

## 常見疑問

**Q: 我需要自己打 `docker build` 或 `docker run` 嗎?**
不用。第 4 步(`02-up-servers.sh`)跟第 6 步(`04-onboard-machine.sh`)裡面都已經包好 build/run 了，你只要跑腳本就好。

**Q: 每次都要重跑全部 10 步嗎?**
不用。開機重來的話，通常只需要:第 1 步(開 Docker Desktop)→ 第 4 步(啟動伺服器，如果沒跑過 `05-teardown.sh` 甚至可以跳過，因為 container 可能還在)→ 第 7、8 步(啟動網頁)。第 3、5、6 步做過一次之後，除非你有跑過 `--purge` 清掉，不然不用重做。
