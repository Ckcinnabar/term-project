# 數據下載指南

## 方法1: Kaggle下載 (推薦)

### 步驟1: 設置Kaggle賬戶
1. 註冊Kaggle賬戶: https://www.kaggle.com/
2. 進入 Account → API → Create New Token
3. 下載kaggle.json文件

### 步驟2: 安裝Kaggle API
```bash
pip install kaggle
```

### 步驟3: 配置API密鑰
```bash
# Windows
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\
```

### 步驟4: 下載PlantVillage數據集
```bash
# 下載完整PlantVillage數據集
kaggle datasets download -d emmarex/plantdisease

# 或下載番茄專用數據集
kaggle datasets download -d charuchaudhry/plantvillage-tomato-leaf-dataset

# 解壓縮
unzip plantdisease.zip
```

## 方法2: 直接從Kaggle網站下載

### PlantVillage完整數據集
- **URL**: https://www.kaggle.com/datasets/emmarex/plantdisease
- **大小**: 約 822 MB
- **內容**: 38個類別，54,303張圖片

### 番茄專用數據集
- **URL**: https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset
- **大小**: 約 147 MB
- **內容**: 10個番茄類別，18,160張圖片

### 下載步驟：
1. 點擊上述連結
2. 點擊 "Download" 按鈕
3. 解壓縮到項目文件夾

## 方法3: GitHub Repository

### PlantVillage GitHub
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
```

## 方法4: Mendeley Data

### Mendeley番茄數據集
- **URL**: https://data.mendeley.com/datasets/ngdgg79rzb/1
- **特點**: 經過預處理的番茄葉子圖像
- **下載**: 需要註冊Mendeley賬戶

## 方法5: TensorFlow Datasets

### 使用TensorFlow直接下載
```python
import tensorflow_datasets as tfds

# 下載PlantVillage數據集
ds = tfds.load('plant_village', split='train', as_supervised=True)

# 保存到本地
tfds.download_and_prepare('plant_village', download_dir='./data/')
```

## 推薦的下載順序

### 第一選擇: Kaggle番茄專用數據集
```bash
kaggle datasets download -d charuchaudhry/plantvillage-tomato-leaf-dataset
```
**理由**:
- 專門針對番茄
- 大小適中 (147 MB)
- 包含所需的Blight類別

### 第二選擇: Kaggle完整數據集
```bash
kaggle datasets download -d emmarex/plantdisease
```
**理由**:
- 數據更全面
- 可以選擇需要的類別

### 第三選擇: 直接網站下載
如果命令行有問題，直接從Kaggle網站下載

## 數據結構預期

下載後的文件夾結構應該是：
```
PlantVillage/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/          # 我們需要的
├── Tomato___Late_blight/           # 我們需要的
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites_Two-spotted_spider_mite/
├── Tomato___Target_Spot/
├── Tomato___Tomato_mosaic_virus/
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
└── Tomato___healthy/
```

## 下載後的驗證

### 檢查數據完整性
```python
import os

def check_dataset(data_path):
    folders = os.listdir(data_path)

    for folder in folders:
        if 'blight' in folder.lower():
            folder_path = os.path.join(data_path, folder)
            image_count = len(os.listdir(folder_path))
            print(f"{folder}: {image_count} images")

# 使用方法
check_dataset("./PlantVillage/")
```

預期輸出：
```
Tomato___Early_blight: 1000 images
Tomato___Late_blight: 1909 images
```

## 遇到問題的解決方案

### 問題1: Kaggle API認證失敗
**解決**: 確保kaggle.json在正確位置，檢查權限

### 問題2: 下載速度慢
**解決**: 使用VPN或選擇GitHub/Mendeley替代源

### 問題3: 文件損壞
**解決**: 重新下載，檢查文件完整性

### 問題4: 存儲空間不足
**解決**: 只下載番茄專用數據集，或選擇部分類別

## 開始下載

現在你可以選擇一種方法開始下載數據了！建議從Kaggle番茄專用數據集開始。