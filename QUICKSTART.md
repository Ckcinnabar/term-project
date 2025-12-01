# 快速開始指南

這個文檔幫助你快速啟動專案並運行 Jupyter notebooks。

## 📋 前置需求

### 硬件
- ✅ CUDA-capable GPU (你已經有了)
- 16+ GB RAM 推薦
- 10+ GB 硬碟空間

### 軟件
- Python 3.8+
- CUDA Toolkit 11.8+ (for PyTorch)
- Conda 或 pip

---

## 🚀 設置環境

### 選項 1: 使用 Conda (推薦)

```bash
# 1. 創建虛擬環境
conda create -n leaf-texture python=3.8
conda activate leaf-texture

# 2. 安裝 PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安裝其他依賴
pip install -r requirements.txt

# 4. 驗證 CUDA 安裝
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 選項 2: 使用 pip

```bash
# 1. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安裝 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝其他依賴
pip install -r requirements.txt

# 4. 驗證 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 檢查數據集

確保你的數據集結構如下：

```
term-project/
├── tomato/
│   ├── train/
│   │   ├── Tomato___Bacterial_spot/  (1000 images)
│   │   ├── Tomato___Early_blight/    (1000 images)
│   │   └── ... (8 more classes)
│   └── val/
│       ├── Tomato___Bacterial_spot/  (100 images)
│       └── ... (9 more classes)
└── notebooks/
```

如果沒有數據集，參考 `Data_Download_Guide.md` 下載。

---

## 🎮 運行 Notebooks

### 步驟 1: 啟動 Jupyter

```bash
# 進入專案目錄
cd term-project

# 啟動 Jupyter Notebook
jupyter notebook
```

瀏覽器會自動打開 http://localhost:8888

### 步驟 2: 按順序運行

| # | Notebook | 說明 | 預計時間 |
|---|----------|------|----------|
| 1 | `01_data_exploration.ipynb` | 探索數據集結構和統計 | 5 分鐘 |
| 2 | `02_preprocessing.ipynb` | 學習圖像預處理流程 | 10 分鐘 |
| 3 | `03_feature_extraction.ipynb` | 提取所有特徵 (需要 GPU) | 30-60 分鐘* |
| 4 | `04_clustering_analysis.ipynb` | PCA 降維和聚類分析 | 15 分鐘 |
| 5 | `05_disease_classification.ipynb` | 訓練疾病分類器 (需要 GPU) | 30-60 分鐘* |
| 6 | `06_dual_application.ipynb` | 完整雙重應用演示 | 10 分鐘 |

\* 時間取決於你處理的圖像數量和 GPU 性能

### 步驟 3: 運行方式

#### 選項 A: 完整運行 (所有數據)
```python
# 在 notebook 中取消註釋這些行
batch_extract_features(TRAIN_DIR, output_path)  # 處理所有 10,000 張
```
⚠️ 需要大量時間和存儲空間

#### 選項 B: 快速測試 (推薦)
```python
# 每個類別只處理 100 張圖像
batch_extract_features(TRAIN_DIR, output_path, limit_per_class=100)
```
✓ 快速驗證流程，1000 張圖像足夠演示

---

## 📊 預期輸出

運行完所有 notebooks 後，你會得到：

### 數據文件
```
term-project/
├── features_train.pkl          # 訓練集特徵 (~1GB 或更少)
├── features_val.pkl            # 驗證集特徵
├── pca_model.pkl               # PCA 模型
├── classifier_model.pth        # 分類器模型
└── notebooks/
    ├── dataset_statistics.csv
    ├── dataset_summary.json
    ├── class_distribution.png
    ├── preprocessing_pipeline.png
    ├── feature_comparison.png
    ├── texture_space_2d.png
    ├── confusion_matrix.png
    └── dual_application_demo.png
```

### 性能指標
- 特徵提取: 1351D 向量/圖像
- PCA 降維: 1351D → 50D (保留 ~95% 方差)
- 聚類: 5 個紋理群組
- 分類準確率: ~90-92% (驗證集)

---

## 🔍 常見問題

### Q1: CUDA out of memory

**解決方案**:
```python
# 減少 batch size
CNN_BATCH_SIZE = 16  # 原本 32

# 或每次處理更少圖像
limit_per_class = 50
```

### Q2: Notebook 運行很慢

**原因**: 可能在使用 CPU 而非 GPU

**檢查**:
```python
import torch
print(torch.cuda.is_available())  # 應該是 True
print(torch.cuda.get_device_name(0))  # 你的 GPU 名稱
```

**如果是 False**: 重新安裝 PyTorch CUDA 版本

### Q3: 找不到圖像文件

**檢查路徑**:
```python
from pathlib import Path
BASE_DIR = Path('..').resolve()
TRAIN_DIR = BASE_DIR / 'tomato' / 'train'
print(TRAIN_DIR.exists())  # 應該是 True
```

### Q4: 想跳過某些 notebooks

**可以跳過**: Notebook 1-2 (如果你已經了解數據集)

**不能跳過**:
- Notebook 3 (特徵提取) - 後續分析需要
- Notebook 5 (分類器訓練) - 雙重應用需要

---

## 🎯 核心功能演示

### 快速測試 (5 分鐘)

如果你只想快速驗證一切正常：

```python
# 測試 CUDA
import torch
print(f"CUDA: {torch.cuda.is_available()}")

# 測試數據載入
from pathlib import Path
TRAIN_DIR = Path('tomato/train')
sample = list(TRAIN_DIR.glob('*/*.jpg'))[0]
print(f"Sample image: {sample}")

# 測試預處理
import cv2
img = cv2.imread(str(sample))
img_resized = cv2.resize(img, (224, 224))
print(f"Resized: {img_resized.shape}")

# 測試 MobileNetV2
from torchvision import models
model = models.mobilenet_v2(pretrained=True)
print("✓ MobileNetV2 loaded")
```

如果以上都成功，你就可以開始運行完整的 notebooks！

---

## 📝 報告撰寫提示

運行完 notebooks 後，使用生成的圖表和數據更新你的 HW4 報告：

### 可以直接使用的圖表
- `class_distribution.png` → Dataset Description
- `preprocessing_pipeline.png` → Methodology: Preprocessing
- `feature_comparison.png` → Methodology: Feature Extraction
- `texture_space_2d.png` → Results: Clustering
- `confusion_matrix.png` → Results: Classification

### 可以引用的數據
- `dataset_statistics.csv` → 表格: 數據集分佈
- `dataset_summary.json` → 文字描述統計數據
- Notebook 輸出的性能指標 → Results 章節

---

## 🆘 需要幫助？

### 錯誤排查順序
1. 檢查 Python 版本: `python --version` (應該 ≥ 3.8)
2. 檢查 CUDA: `nvidia-smi` (查看 GPU 狀態)
3. 檢查 PyTorch: `python -c "import torch; print(torch.__version__)"`
4. 檢查數據集路徑: 確保 `tomato/` 資料夾存在

### 聯繫資訊
- GitHub Issues: [報告問題]
- README.md: 詳細文檔
- notebooks/README_NOTEBOOKS.md: Notebook 詳細說明

---

## ✅ 檢查清單

開始前確認:
- [ ] Python 3.8+ 已安裝
- [ ] CUDA GPU 可用
- [ ] PyTorch with CUDA 已安裝
- [ ] 數據集已下載到 `tomato/` 目錄
- [ ] requirements.txt 所有依賴已安裝
- [ ] Jupyter Notebook 可以啟動

準備運行:
- [ ] 已閱讀 README.md
- [ ] 已閱讀 notebooks/README_NOTEBOOKS.md
- [ ] 了解每個 notebook 的功能
- [ ] 決定使用完整數據集或快速測試

完成後:
- [ ] 所有 notebooks 成功運行
- [ ] 特徵文件已生成
- [ ] 分類器模型已訓練
- [ ] 圖表已保存
- [ ] 準備撰寫報告

---

**祝你實驗順利！ 🎉**
