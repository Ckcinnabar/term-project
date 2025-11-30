# PlantVillage數據集實現計劃

## 數據預處理

### 1. 數據組織
```
PlantVillage/
├── Tomato_Bacterial_spot/     (2127張)
├── Tomato_Early_blight/       (1000張)
├── Tomato_Late_blight/        (1909張)
├── Tomato_Leaf_Mold/          (952張)
├── Tomato_Septoria_leaf_spot/ (1771張)
├── Tomato_Spider_mites/       (1676張)
├── Tomato_Target_Spot/        (1404張)
├── Tomato_Mosaic_virus/       (373張)
├── Tomato_Yellow_Leaf_Curl/   (5357張)
└── Tomato_healthy/            (1591張)
```

### 2. 特徵提取
- **顏色特徵**: RGB直方圖、HSV顏色空間
- **紋理特徵**: GLCM、LBP (Local Binary Pattern)
- **形狀特徵**: 邊緣檢測、輪廓特徵
- **統計特徵**: 均值、方差、偏度、峰度

## 組件1: Classification (分類)

### 分類策略重新設計：

#### 選項A: Blight疾病程度分類 (推薦)
專注於Early Blight和Late Blight，根據病斑面積和嚴重程度分類：
- **輕度Blight** (10-30%葉面受影響)
- **中度Blight** (30-60%葉面受影響)
- **重度Blight** (60%+葉面受影響)

#### 選項B: 疾病類型分類
- **真菌性疾病**: Early blight, Late blight, Leaf Mold, Septoria leaf spot
- **病毒性疾病**: Mosaic virus, Yellow Leaf Curl virus
- **蟲害**: Spider mites
- **細菌性疾病**: Bacterial spot
- **健康**: Healthy

#### 選項C: 疾病vs健康二分類
- **健康葉子**
- **患病葉子** (所有疾病合併)

### 實現方法 (以Blight程度分類為例)：

1. **數據標註**：
```python
# 根據視覺病斑面積手動標註或使用圖像分析
blight_severity = {
    'mild': [],      # 輕度病斑
    'moderate': [],  # 中度病斑
    'severe': []     # 重度病斑
}
```

2. **特徵提取**：
- 病斑區域面積比例
- 病斑顏色特徵 (棕色/黑色斑點)
- 葉子邊緣完整性
- 紋理粗糙度

### 算法實現：
1. **Support Vector Machine (SVM)**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **Logistic Regression**

### 評估指標：
- Accuracy, Precision, Recall, F1-score
- 混淆矩陣

## 組件2: Prediction/Regression (預測)

### 創建回歸標籤：
```python
# 疾病嚴重程度評分
disease_severity = {
    'Tomato_healthy': 0,
    'Tomato_Spider_mites': 25,
    'Tomato_Target_Spot': 35,
    'Tomato_Mosaic_virus': 45,
    'Tomato_Leaf_Mold': 55,
    'Tomato_Septoria_leaf_spot': 65,
    'Tomato_Yellow_Leaf_Curl': 70,
    'Tomato_Bacterial_spot': 80,
    'Tomato_Early_blight': 85,
    'Tomato_Late_blight': 95
}
```

### 回歸算法：
1. **Linear Regression**
2. **Polynomial Regression** (度數2-3)
3. **Ridge Regression** (α調優)
4. **Lasso Regression** (特徵選擇)
5. **Support Vector Regression (SVR)**

### 評估指標：
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## 組件3: Clustering (聚類)

### 聚類策略：

#### 策略A: 疾病症狀相似性
- 提取所有圖片的特徵
- 使用K-means找出視覺相似的疾病群組
- 分析每個群組的疾病分布

#### 策略B: 疾病嚴重程度分組
- 在每種疾病內部進行聚類
- 發現輕度/中度/重度症狀模式

#### 策略C: 跨疾病模式發現
- 找出不同疾病間的共同特徵
- 識別易混淆的疾病組合

### 聚類算法：
1. **K-Means** (k=3,5,8,10)
2. **Hierarchical Clustering** (Ward linkage)
3. **DBSCAN** (密度聚類)

### 評估指標：
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

## 三個組件的統一設計

### 核心研究問題：
**如何通過葉片表面特徵分析Blight疾病的嚴重程度？**

1. **Classification**: Blight嚴重程度分類 (輕度/中度/重度)
2. **Regression**: 預測病斑覆蓋面積百分比 (0-100%)
3. **Clustering**: 發現不同嚴重程度的視覺模式

### 統一標註策略：
```python
# 針對Early Blight和Late Blight圖像
for image in blight_images:
    affected_area_percent = calculate_affected_area(image)

    # Classification標籤
    if affected_area_percent < 30:
        severity_class = 'mild'
    elif affected_area_percent < 60:
        severity_class = 'moderate'
    else:
        severity_class = 'severe'

    # Regression標籤
    severity_score = affected_area_percent
```

## 實現順序

### Phase 1: 數據準備
1. 下載PlantVillage數據集
2. 篩選Early Blight和Late Blight圖像
3. Blight嚴重程度標註
4. 特徵提取

### Phase 2: 分類實現 (必需)
1. Blight嚴重程度三分類
2. 實現傳統ML算法
3. 性能評估

### Phase 3: 回歸實現 (必需)
1. 病斑面積百分比預測
2. 實現回歸算法
3. 回歸性能評估

### Phase 4: 聚類實現 (必需)
1. 實現3種聚類算法
2. 結果可視化分析
3. 聚類質量評估

### Phase 5: CNN比較 (加分)
1. 實現CNN模型
2. 與傳統ML比較
3. 性能分析報告