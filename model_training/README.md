# Split Data
> python ./split.py --target_data `../ROSPaCe_complete/ROSPaCe_complete.csv ` --split_dir `./split` --unit `1000` --chunk `1` --mode `Direct`

## 切分方法
### 直接切割（Direct）
不打亂原始資料，直接依據資料在母體中的先後順序或索引大小進行區段切割。
複製一份到tmp folder -> 切割unit大小出去 -> 分成chunk數量的資料 -> 切出去的資料請不要在tmp中存在（刪除）

### 隨機切割（Random）
機率性抽取和打亂（shuffle）。
複製一份到tmp folder -> 切割unit大小出去 -> 隨機抽換資料（shuffle） -> 切出去的資料請不要在tmp中存在（刪除）

### 分層抽樣（SR, Stratified Resampling）
讓各資料集有特定主導類別（80%），20% 的其他類別強制規定其餘個類別在 20% 中均分（各佔 4%），或是根據少數類別的比例進行過採樣（Oversampling）。
過採樣方法：
 - Random Oversampling：直接複製並放回少數類別的樣本。
 - SMOTE：據少數類別樣本的特徵空間，利用鄰近點合成新樣本。
複製一份到tmp folder -> 建立index table -> 在類別內隨機切分 -> 切出去的資料請不要在tmp中存在（刪除）

### 狄利克雷分布切分（DD, Dirichlet Distribution）
不採用硬性比例，使用狄利克雷分佈 \(\text{Dirichlet}(\alpha)\) 作為隨機向量生成器，來控制每個類別分配到各資料集的比例。
複製一份到tmp folder -> 建立index table -> DD計算分布 -> 透過分布在類別內隨機切分 -> 切出去的資料請不要在tmp中存在（刪除）