# Split Data

`split` 模組將 CSV 資料切分成多個 chunk。所有切分方法都繼承 `SplitBase`，使用相同的暫存資料、index table、輸出格式與清理流程。

## 共用流程

每次建立 splitter 時，會依序執行以下步驟：

1. **建立 index table**
	- 檢查來源 CSV 是否存在。
	- 在 `tmp_dir` 建立暫存 CSV，並複製來源資料。
	- 讀取 CSV header，尋找 `attack` 欄位。
	- 逐行記錄每筆資料的 byte offset、byte length 與 `attack` 類別。
	- 後續輸出資料時直接依 offset 讀取，不需要重新解析整個 CSV。
2. **建立輸出設定**
	- 建立 `output_dir` 與 `chart_dir`。
	- 建立 `chunk_num` 個輸出檔案：`chunk_0.csv`、`chunk_1.csv`，依此類推。
	- 每個 chunk 的目標大小由 `chunk_size` 決定。
3. **選擇本輪資料**
	- 各切分方法依自己的規則從 index table 選取資料。
	- 每個 chunk 都會寫入原始 header。
4. **移除已切出的資料**
	- 從 index table 移除本輪已輸出的 records。
	- 將尚未使用的 records 重寫回暫存 CSV。
	- 重新建立 index table，供下一輪切分使用。
5. **清理暫存資料**
	- 所有輪次完成後呼叫 `splitter.cleanup()`，刪除暫存 CSV。

當剩餘資料不足以填滿所有 chunk 時，模組會平均分配剩餘資料，餘數放在最後一個 chunk。來源 CSV 必須包含 `attack` 欄位。

## 切分方法

### 直接切割（Sequential）

依照資料在暫存 CSV 中的原始順序切割，不進行隨機化。

具體步驟：

1. 建立 index table。
2. 從 records 的開頭取出第一個 chunk 所需的資料。
3. 接續取出其他 chunks，直到完成所有 chunk。
4. 將取出的 records 寫入輸出檔案。
5. 移除已取出的 records，重寫剩餘資料。

### 隨機切割（Random）

不考慮 `attack` 類別比例，將目前剩餘的 records 完整洗牌後依序分配。

具體步驟：

1. 建立 index table。
2. 複製目前 records 並隨機打亂。
3. 依各 chunk 容量，從洗牌後的 records 依序取出資料。
4. 將每個 chunk 寫入輸出檔案。
5. 移除已取出的 records，重寫剩餘資料。

設定 `random_seed` 可以讓相同輸入得到可重現的切分結果。

```python
from split import RandomSplit

splitter = RandomSplit(
	 src_path="source.csv",
	 tmp_dir="tmp",
	 output_dir="split",
	 chart_dir="img",
	 chunk_size=1000,
	 chunk_num=5,
	 random_seed=42,
)
splitter.split_to_chunks()
splitter.cleanup()
```

### 分層抽樣（SR, Stratified Resampling）

以 `attack` 欄位作為分層依據。每一層會依目前剩餘資料中的類別比例抽樣，再將抽出的資料隨機分配到各 chunk。

具體步驟：

1. 建立 index table，並依 `attack` 類別將 records 分組。
2. 在每個類別內分別洗牌。
3. 依所有類別在剩餘資料中的比例，計算本輪各類別應抽出的數量。
4. 將各類別抽出的 records 隨機放入尚有容量的 chunks。
5. 將各 chunk 的 records 再次洗牌並寫入輸出檔案。
6. 移除已取出的 records，重寫剩餘資料。

目前 SR 實作是**依比例抽樣**，尚未實作固定 80/20 比例、Random Oversampling 或 SMOTE。

```python
from split import StratifiedSplit

splitter = StratifiedSplit(
	 src_path="source.csv",
	 tmp_dir="tmp",
	 output_dir="split",
	 chart_dir="img",
	 chunk_size=1000,
	 chunk_num=5,
	 random_seed=42,
)
splitter.split_to_chunks()
splitter.cleanup()
```

### 狄利克雷分布切分（DD, Dirichlet Distribution）

以 `attack` 欄位分層，但不使用固定的類別比例。每個類別會獨立產生一組 Dirichlet 權重，讓不同 chunk 可能擁有不同的類別分布。

具體步驟：

1. 建立 index table，並依 `attack` 類別將 records 分組。
2. 在每個類別內洗牌。
3. 使用 `alpha` 參數產生各 chunk 的 Gamma 權重；將權重正規化後，即得到 Dirichlet 比例。
4. 依該類別的 Dirichlet 比例計算各 chunk 的分配數量。
5. 若某 chunk 已達容量，將剩餘數量依其他 chunk 的可用容量重新分配。
6. 將所有類別資料寫入輸出檔案並打亂 chunk 內順序。
7. 移除已取出的 records，重寫剩餘資料。

`alpha` 必須大於 0：

- `alpha < 1`：分布較集中，類別可能集中在少數 chunks。
- `alpha = 1`：接近均勻的隨機分布。
- `alpha > 1`：分布較平均。

```python
from split import DirichletSplit

splitter = DirichletSplit(
	 src_path="source.csv",
	 tmp_dir="tmp",
	 output_dir="split",
	 chart_dir="img",
	 chunk_size=1000,
	 chunk_num=5,
	 alpha=0.5,
	 random_seed=42,
)
splitter.split_to_chunks()
splitter.cleanup()
```

## 參數

| 參數 | 說明 |
| --- | --- |
| `src_path` | 來源 CSV 路徑，必須包含 `attack` 欄位。 |
| `tmp_dir` | 暫存 CSV 所在目錄。 |
| `output_dir` | chunk 輸出目錄。 |
| `chart_dir` | 圖表輸出目錄，目前只由 base 建立目錄。 |
| `chunk_size` | 每個 chunk 的目標資料筆數。 |
| `chunk_num` | 本輪要產生的 chunk 數量。 |
| `random_seed` | Random、SR、DD 的可重現隨機種子。 |
| `alpha` | DD 的 Dirichlet 濃度參數，必須大於 0。 |