# Regression實現詳細計劃

## 回歸目標：預測Blight病斑覆蓋面積百分比 (0-100%)

### 1. 創建回歸標籤 (Ground Truth)

#### 方法A: 半自動圖像分析 (推薦)
```python
import cv2
import numpy as np

def calculate_disease_area_percentage(image_path):
    # 1. 讀取圖像
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 分離健康葉子區域 (綠色)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 3. 分離病斑區域 (棕色/黃色/黑色)
    # Blight通常呈現棕色到黑色的病斑
    disease_lower = np.array([10, 50, 20])
    disease_upper = np.array([30, 255, 200])
    disease_mask = cv2.inRange(hsv, disease_lower, disease_upper)

    # 4. 計算面積百分比
    total_leaf_area = np.sum(green_mask > 0) + np.sum(disease_mask > 0)
    disease_area = np.sum(disease_mask > 0)

    if total_leaf_area > 0:
        percentage = (disease_area / total_leaf_area) * 100
        return min(percentage, 100)  # 限制在100%以內
    return 0
```

#### 方法B: 手動標註 (更準確但耗時)
```python
# 使用標註工具為每張圖片手動標記病斑面積
manual_annotations = {
    'early_blight_001.jpg': 25.5,  # 25.5%病斑覆蓋
    'early_blight_002.jpg': 45.2,  # 45.2%病斑覆蓋
    'late_blight_001.jpg': 67.8,   # 67.8%病斑覆蓋
    # ... 更多標註
}
```

#### 方法C: 基於分類的映射 (簡化版)
```python
# 基於PlantVillage現有分類創建粗略估計
severity_mapping = {
    'Tomato_healthy': 0,
    'Tomato_Early_blight': np.random.normal(40, 15),  # 40±15%
    'Tomato_Late_blight': np.random.normal(70, 20),   # 70±20%
}
```

### 2. 特徵工程 (針對回歸優化)

#### 顏色特徵
```python
def extract_color_features(image):
    # RGB通道統計
    r_mean, g_mean, b_mean = np.mean(image, axis=(0,1))
    r_std, g_std, b_std = np.std(image, axis=(0,1))

    # HSV通道統計
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0,1))

    # 病斑相關的顏色特徵
    brown_ratio = calculate_brown_pixel_ratio(image)
    yellow_ratio = calculate_yellow_pixel_ratio(image)

    return [r_mean, g_mean, b_mean, r_std, g_std, b_std,
            h_mean, s_mean, v_mean, brown_ratio, yellow_ratio]
```

#### 紋理特徵
```python
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern

def extract_texture_features(gray_image):
    # GLCM紋理特徵
    glcm = greycomatrix(gray_image, [1], [0, 45, 90, 135])
    contrast = greycoprops(glcm, 'contrast').mean()
    dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
    homogeneity = greycoprops(glcm, 'homogeneity').mean()
    energy = greycoprops(glcm, 'energy').mean()

    # LBP特徵
    lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=10)[0]

    # 邊緣密度 (病斑邊界特徵)
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return [contrast, dissimilarity, homogeneity, energy,
            edge_density] + lbp_hist.tolist()
```

#### 形狀和幾何特徵
```python
def extract_shape_features(image):
    # 病斑區域的形狀特徵
    disease_regions = segment_disease_areas(image)

    features = []
    for region in disease_regions:
        # 病斑大小
        area = cv2.contourArea(region)
        perimeter = cv2.arcLength(region, True)

        # 形狀複雜度
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

        # 病斑分布
        moments = cv2.moments(region)

        features.extend([area, perimeter, compactness])

    return features
```

### 3. 回歸算法實現

#### Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 基本線性回歸
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

#### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures

# 多項式特徵擴展
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
```

#### Ridge Regression (L2正則化)
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 參數調優
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_
```

#### Support Vector Regression (SVR)
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 特徵標準化 (SVR需要)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR實現
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train_scaled, y_train)
```

#### Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_reg.fit(X_train, y_train)

# 特徵重要性分析
feature_importance = rf_reg.feature_importances_
```

### 4. 評估指標

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_regression_model(y_true, y_pred, model_name):
    # 計算評估指標
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Results:")
    print(f"MAE: {mae:.2f}%")
    print(f"RMSE: {rmse:.2f}%")
    print(f"R²: {r2:.3f}")

    # 可視化預測vs實際
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('實際病斑面積 (%)')
    plt.ylabel('預測病斑面積 (%)')
    plt.title(f'{model_name} - 預測 vs 實際')
    plt.show()

    return mae, rmse, r2
```

### 5. 實施步驟

#### Phase 1: 數據準備
1. 篩選Early Blight和Late Blight圖像
2. 使用圖像分析計算病斑面積百分比
3. 手動檢查和調整部分標註

#### Phase 2: 特徵提取
1. 實現顏色、紋理、形狀特徵提取
2. 特徵標準化和選擇
3. 創建完整的特徵矩陣

#### Phase 3: 模型訓練
1. 實現5種回歸算法
2. 交叉驗證和超參數調優
3. 模型性能比較

#### Phase 4: 結果分析
1. 錯誤分析 (哪些圖像預測困難)
2. 特徵重要性分析
3. 與分類結果的一致性檢查

### 6. 預期結果

#### 良好性能指標：
- **MAE < 10%**: 平均絕對誤差小於10個百分點
- **RMSE < 15%**: 均方根誤差小於15個百分點
- **R² > 0.7**: 解釋變異大於70%

#### 挑戰和解決方案：
- **標註困難**: 使用多種方法交叉驗證
- **特徵選擇**: 使用特徵重要性和相關性分析
- **過擬合**: 使用正則化和交叉驗證