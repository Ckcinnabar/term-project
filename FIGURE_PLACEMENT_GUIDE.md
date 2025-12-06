# IEEE Paper - Figure Placement Guide

## 📊 Complete Figure List (15 Figures)

---

## SECTION III - METHODOLOGY

### **Figure 1**: Sample Images
- **File**: `notebooks/sample_images.png`
- **Location**: Section III.A - Dataset
- **Caption**: Representative samples from each of the 10 classes in the PlantVillage tomato leaf dataset.
- **Purpose**: Show visual differences between disease categories

### **Figure 2**: Class Distribution
- **File**: `notebooks/class_distribution.png`
- **Location**: Section III.A - Dataset
- **Caption**: Dataset distribution showing balanced class representation across training and validation sets.
- **Purpose**: Demonstrate balanced dataset (important for fair evaluation)

### **Figure 3**: Color Distribution
- **File**: `notebooks/color_distribution.png`
- **Location**: Section III.B.2 - Background Removal
- **Caption**: RGB channel distributions showing dominant green channel (mean: 116.7) consistent with natural leaf imagery.
- **Purpose**: Validate preprocessing and show data characteristics

### **Figure 4**: Feature Comparison
- **File**: `notebooks/feature_comparison.png`
- **Location**: Section III.C.4 - Deep Learning Features
- **Caption**: Comparison of feature extraction methods showing complementary information from GLCM, fractal, vein, and CNN features.
- **Purpose**: Illustrate multi-modal feature extraction approach

### **Figure 5**: PCA Variance
- **File**: `notebooks/pca_variance.png`
- **Location**: Section III.D - Dimensionality Reduction
- **Caption**: (Left) Scree plot showing variance contribution per component. (Right) Cumulative explained variance reaching 70.74% with 50 components.
- **Purpose**: Justify PCA component selection

### **Figure 6**: Optimal K Selection
- **File**: `notebooks/optimal_k_selection.png`
- **Location**: Section III.F - Application 2 (Automatic K Selection)
- **Caption**: Automatic K selection using four complementary metrics. Red line indicates selected optimal k.
- **Purpose**: Show algorithm for automatic cluster number determination (KEY CONTRIBUTION)

### **Figure 7**: Dendrogram
- **File**: `notebooks/dendrogram.png`
- **Location**: Section III.F - Hierarchical Clustering
- **Caption**: Hierarchical clustering dendrogram showing multi-scale texture relationships and cluster merging sequence.
- **Purpose**: Validate K-means with hierarchical approach

---

## SECTION V - RESULTS AND DISCUSSION

### **Figure 8**: Training History
- **File**: `notebooks/training_history.png`
- **Location**: Section V.A - Disease Classification Performance
- **Caption**: Training curves showing (Left) loss convergence and (Right) accuracy progression. Best validation accuracy of 99.40% achieved at epoch 19.
- **Purpose**: Show training stability and convergence

### **Figure 9**: Confusion Matrix
- **File**: `notebooks/confusion_matrix.png`
- **Location**: Section V.A - Per-Class Results
- **Caption**: Confusion matrix showing high diagonal values with minimal misclassification. Spider Mites and Mosaic Virus achieve perfect 100% accuracy.
- **Purpose**: Detailed per-class performance analysis (KEY RESULT)

### **Figure 10**: Texture Space 2D
- **File**: `notebooks/texture_space_2d.png`
- **Location**: Section V.B - Cluster Characteristics
- **Caption**: 2D visualization of texture space using first two principal components. (Left) K-means clusters with centroids. (Right) Hierarchical clustering comparison.
- **Purpose**: Visualize cluster separation in reduced space

### **Figure 11**: Texture Space 3D
- **File**: `notebooks/texture_space_3d.png`
- **Location**: Section V.B - Cluster Characteristics
- **Caption**: 3D texture space visualization using PC1, PC2, PC3, showing cluster separation in reduced feature space.
- **Purpose**: Enhanced 3D perspective of clustering

### **Figure 12**: Cluster vs Disease
- **File**: `notebooks/cluster_vs_disease.png`
- **Location**: Section V.C - Cross-Application Analysis
- **Caption**: Heatmap showing correlation between cluster assignments (texture-based) and original disease labels (visual diagnosis).
- **Purpose**: Show disease-texture correlation (KEY FINDING)

### **Figure 13**: Disease Texture Features
- **File**: `notebooks/disease_texture_features.png`
- **Location**: Section V.C - Correlation Findings
- **Caption**: Box plots showing distribution of key texture features (contrast, fractal dimension, vein density) across disease categories.
- **Purpose**: Statistical analysis of texture parameters per disease

### **Figure 14**: Disease Texture Association
- **File**: `notebooks/disease_texture_association.png`
- **Location**: Section V.C - Correlation Findings
- **Caption**: Scatter plot showing relationship between disease severity (measured by CNN confidence) and texture roughness parameters.
- **Purpose**: Quantify disease-texture relationship

### **Figure 15**: Dual Application Demo
- **File**: `notebooks/dual_application_demo.png`
- **Location**: Section V.D - Dual-Application Framework Validation
- **Caption**: Side-by-side comparison demonstrating dual outputs: (Left) Agricultural classification with disease label and confidence, (Right) Engineering analysis with quantitative surface parameters.
- **Purpose**: Illustrate dual framework concept (CORE CONTRIBUTION)

---

## 📝 Figure Organization Summary

### By Section:
- **Section III (Methodology)**: 7 figures (Figures 1-7)
- **Section V (Results)**: 8 figures (Figures 8-15)

### By Type:
- **Dataset/Preprocessing**: 3 figures (1, 2, 3)
- **Feature Extraction**: 2 figures (4, 5)
- **Clustering Method**: 2 figures (6, 7)
- **Classification Results**: 2 figures (8, 9)
- **Clustering Results**: 2 figures (10, 11)
- **Cross-Domain Analysis**: 4 figures (12, 13, 14, 15)

---

## ⚡ Quick Checklist for Paper Submission

### Before Submission:
- [ ] All 15 figures placed in correct sections
- [ ] Figure captions match the markdown exactly
- [ ] Figure numbers sequential (1-15)
- [ ] High resolution (300 DPI minimum for IEEE)
- [ ] Color figures acceptable (check journal requirements)
- [ ] Figure file formats: PNG (preferred) or EPS/PDF for publication

### Figure Quality Check:
```bash
# Check all figures exist
ls notebooks/*.png

# Expected 15 files:
# 1. sample_images.png
# 2. class_distribution.png
# 3. color_distribution.png
# 4. feature_comparison.png
# 5. pca_variance.png
# 6. optimal_k_selection.png
# 7. dendrogram.png
# 8. training_history.png
# 9. confusion_matrix.png
# 10. texture_space_2d.png
# 11. texture_space_3d.png
# 12. cluster_vs_disease.png
# 13. disease_texture_features.png
# 14. disease_texture_association.png
# 15. dual_application_demo.png
```

---

## 🎨 IEEE Figure Formatting Guidelines

### Size Requirements:
- **Single column**: 3.5 inches (8.89 cm) wide
- **Double column**: 7.16 inches (18.19 cm) wide
- **Height**: Max 9.0 inches (22.86 cm)

### Resolution:
- **Minimum**: 300 DPI
- **Preferred**: 600 DPI for line art
- **Photos/Grayscale**: 300 DPI acceptable

### Font in Figures:
- **Minimum size**: 8 pt
- **Preferred**: 10-12 pt
- **Font family**: Times, Helvetica, or Arial

### Color:
- IEEE allows color figures (online and print)
- Ensure figures readable in grayscale (some readers print B&W)
- Use colorblind-friendly palettes

---

## 💡 Tips for Final Paper Preparation

1. **Convert Markdown to IEEE LaTeX**:
   - Use IEEE template: https://www.ieee.org/conferences/publishing/templates.html
   - Template class: `\documentclass[conference]{IEEEtran}`

2. **Figure References in Text**:
   - Always refer to figures in text: "as shown in Fig. 1..."
   - Place figures close to first reference

3. **Multi-Part Figures**:
   - Use (a), (b), (c) labels within figures
   - Example: Fig. 5 has (Left) and (Right) - label as (a) and (b)

4. **Figure Captions**:
   - IEEE style: "Fig. 1. Caption text here."
   - Period after figure number
   - Capitalize "Fig."

5. **Tables**:
   - Use "TABLE I", "TABLE II" (Roman numerals)
   - Caption goes ABOVE table (opposite of figures)

---

## 📄 LaTeX Figure Template

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=3.5in]{notebooks/sample_images.png}
\caption{Representative samples from each of the 10 classes in the PlantVillage tomato leaf dataset.}
\label{fig:sample_images}
\end{figure}
```

For double-column figures:
```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=7in]{notebooks/texture_space_2d.png}
\caption{2D visualization of texture space using first two principal components. (a) K-means clusters with centroids. (b) Hierarchical clustering comparison.}
\label{fig:texture_2d}
\end{figure*}
```

---

## ✅ Final Verification

| Figure | File Exists | Resolution OK | Caption Match | Referenced in Text |
|--------|-------------|---------------|---------------|-------------------|
| Fig. 1 | ✓ | Check | Check | Check |
| Fig. 2 | ✓ | Check | Check | Check |
| Fig. 3 | ✓ | Check | Check | Check |
| ... | ... | ... | ... | ... |

Use this checklist before submission!
