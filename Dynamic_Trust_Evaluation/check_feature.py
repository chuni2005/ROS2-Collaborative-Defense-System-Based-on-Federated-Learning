import pandas as pd

def print_feature_list():
    excel_file = 'feature_list.xlsx'
    
    try:
        print(f"正在讀取 {excel_file} ...\n")
        # 讀取 'features_list' 工作表
        df = pd.read_excel(excel_file, sheet_name='features_list')
        
        # 萃取特徵名稱 (feature_name) 欄位並轉為 List
        features = df['feature_name'].dropna().tolist()
        
        print(f"✅ 成功讀取！總共找到 {len(features)} 個特徵：")
        print("-" * 40)
        
        # 逐一印出特徵
        for i, feature in enumerate(features, 1):
            print(f"{i:3d}. {feature}")
            
        print("-" * 40)
        print("印出完畢。")
            
    except FileNotFoundError:
        print(f"❌ 找不到檔案 '{excel_file}'，請確認檔案與這支程式在同一個資料夾。")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    print_feature_list()