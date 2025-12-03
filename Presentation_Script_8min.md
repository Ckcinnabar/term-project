# LeafInspire: 8-Minute Presentation Script

**Total Time: ~8 minutes (480 seconds)**

---

## [Slide 1: Title Slide] (15 seconds)

Good morning everyone. My name is Kuan-Chen Chen, and today I'm excited to present **LeafInspire** – a dual-application framework for tomato disease detection and biomimetic surface analysis using deep learning.

This project demonstrates how a single plant image dataset can serve two completely different domains: agriculture and engineering.

---

## [Slide 2: Introduction] (30 seconds)

Let me start with the motivation. Plant diseases significantly reduce crop yields, and farmers urgently need fast and reliable detection tools. At the same time, natural organisms – especially leaf textures – have long inspired biomimetic material design in engineering.

However, most research uses image datasets for only a single purpose. This raises an important question: **Can one dataset serve multiple domains at once?**

This project proves that plant imagery can support both agricultural diagnosis and engineering texture modeling simultaneously. This dual-application approach maximizes the value we extract from existing datasets.

---

## [Slide 3: Dataset Overview] (30 seconds)

We use the PlantVillage tomato leaf dataset from Kaggle, which contains approximately 12,000 labeled high-resolution images across 10 classes: 9 different diseases plus healthy leaves.

The dataset was captured under controlled lighting with uniform backgrounds, ensuring high quality. It covers a wide variety of biological symptoms – from curling and blistering patterns to color chlorosis, necrotic lesions, and mold-like textures.

This clear visual diversity is crucial because it allows robust feature learning across multiple tasks, making it ideal for our dual-application framework.

---

## [Slide 4: Dataset Statistics & Properties] (25 seconds)

Looking at the dataset properties: we have a balanced class distribution across both training and validation sets. This prevents class dominance and reduces label imbalance bias, ensuring stable optimization during CNN training.

The color distribution analysis shows consistent illumination across images, with the green channel slightly dominant – which aligns perfectly with natural leaf pigmentation. This confirms the reliability of our preprocessing pipeline.

---

## [Slide 5: Preprocessing Pipeline] (30 seconds)

Our preprocessing follows a six-step workflow:

First, we resize all images to 224×224 for uniformity. Second, we apply median filtering to remove noise while maintaining structural edges. Third, we convert to HSV color space for more stable color-based segmentation. Fourth, we use saturation thresholding to remove the background. Fifth, we extract the leaf foreground mask. Finally, we perform grayscale transformation for texture extraction.

The purpose is threefold: standardize input for both CNN and texture analysis, remove irrelevant background textures, and improve GLCM and fractal feature accuracy.

---

## [Slide 6: Preprocessing Comparison] (15 seconds)

This slide shows cross-class robustness. Our segmentation works even on difficult cases – light-colored backgrounds, damaged leaf edges. Importantly, mold-like diseases maintain shape integrity after filtering, and background removal successfully isolates disease-related texture patterns. This is essential for stable feature computation across all disease types.

---

## [Slide 7: Feature Extraction] (35 seconds)

Now, the core of our approach: multi-source feature extraction. We combine classical texture features with deep learning features to create a comprehensive 1351-dimensional feature vector.

For classical features: GLCM provides 60 dimensions capturing spatial co-occurrence of pixel intensities – sensitive to roughness, lesion granularity, and patch irregularity. Fractal dimension adds 1 dimension measuring multi-scale complexity. Vein features contribute 10 dimensions including vein density, geometric compactness, and Hu moments.

For deep learning: MobileNetV2 extracts 1280 dimensions combining semantic and texture features. It's pretrained on ImageNet, giving us a strong general feature base, and it's lightweight and efficient – perfect for deployment.

---

## [Slide 8: PCA Dimensionality Reduction] (20 seconds)

To handle this high-dimensional data, we apply PCA dimensionality reduction. After standardization with StandardScaler, PCA reduces our 1351 dimensions down to just 50 principal components, while retaining approximately 95% of cumulative variance.

This eliminates redundant features and enhances clustering stability. The scree plot shows that the first few components capture the dominant texture patterns, revealing natural groupings among leaf textures.

---

## [Slide 9: Training Progress] (20 seconds)

For disease classification, we fine-tuned MobileNetV2. The training dynamics show rapid decrease in training loss within the first 5 epochs, while validation loss remains stable – indicating minimal overfitting. Accuracy quickly converges above 95%.

Note the sharp accuracy dip at epoch 6, which suggests temporary over-regularization. However, the model recovers and stabilizes, reaching a final accuracy near 99%. This demonstrates strong generalization ability.

---

## [Slide 10: Classification Performance] (25 seconds)

Our final classification performance is exceptional: 99.4% validation accuracy. The confusion matrix shows nearly perfect separation across all 10 categories, with misclassifications extremely rare – less than 0.5%.

Even diseases with similar early-stage symptoms, like Target Spot versus Late blight, are classified correctly. This confirms the CNN's ability to extract highly discriminative features, making it reliable for real-world agricultural applications.

---

## [Slide 11: Selecting k for Clustering] (20 seconds)

Moving to the engineering application: texture clustering. We used both the elbow method and silhouette score to determine the optimal number of clusters.

The elbow method shows a clear elbow around k=5, where inertia drops quickly between k=2 and k=5, then plateaus. The silhouette score reaches its highest local value near k=5, indicating moderate cluster separation in texture space.

Therefore, we selected k=5 for the best balance between compactness and interpretability.

---

## [Slide 12-13: Texture Space Visualization] (20 seconds)

These visualizations show our texture space in both 2D and 3D. The K-means clustering on the left shows five distinct texture clusters with their centroids marked. The hierarchical clustering on the right provides an alternative view.

You can clearly see natural groupings emerging from the PCA-reduced feature space. Different colors represent different clusters, and the spatial distribution reveals how diseases with similar texture characteristics group together.

---

## [Slide 14: Disease–Cluster Distribution] (45 seconds)

This heatmap reveals fascinating disease-cluster relationships. Diseases are strongly aligned with specific texture groups.

Let me walk through all five clusters: **Cluster 0**, with 114 samples, shows high roughness at 505 – this is a sandblasted surface analog, dominated by Tomato mosaic virus. **Cluster 1**, the largest at 129 samples, has medium roughness at 220 – a honed surface with directional grooves, mostly healthy leaves. **Cluster 2**, 139 samples, exhibits the highest roughness at 795 – also a sandblasted analog, primarily Target Spot disease. **Cluster 3**, 114 samples, shows high roughness at 488 and dense vein structure – sandblasted texture, dominated by Spider mites. Finally, **Cluster 4**, only 4 samples, has the lowest roughness at 171 with sparse veins – a dimpled texture, exclusively Leaf Mold.

The key insight here is that even without disease labels, texture features alone reflect biological patterns. This validates our engineering application.

---

## [Slide 15: Texture Feature Distribution] (40 seconds)

Now let's dive into the corrected texture feature analysis. We analyzed four engineered metrics across all diseases:

**Roughness**: Highest in Late blight at 812.98, lowest in Bacterial spot at 197.66. High roughness correlates with severe lesion formation.

**Anisotropy**: Leaf Mold shows the strongest directional texture at 0.00109, indicating highly organized fungal growth patterns.

**Complexity**, measured by fractal dimension: Highest in Late blight at 2.087 and Tomato mosaic virus at 2.081, reflecting irregular disease progression patterns.

**Vein Density**: Highest in Tomato mosaic virus at 0.306, lowest in Leaf Mold at 0.229. Viral infections preserve vein structure better than fungal diseases.

The key takeaway: each disease exhibits a distinct texture signature, supporting meaningful clustering and cross-domain interpretation.

---

## [Slide 16: Disease–Cluster Relationships] (25 seconds)

Looking at cross-domain patterns: each disease has a signature texture profile, and these strong correlations suggest exciting possibilities.

First, potential for disease pre-screening via texture alone – before traditional diagnostic tests. Second, texture-based engineering applications inspired by biological surfaces.

For engineering relevance: rough surfaces can enhance friction, dense vein patterns can inform micro-channel flow modeling, and complex surfaces have applications in light scattering and diffusion systems.

---

## [Slide 17: Full Dual-Application Demonstration] (30 seconds)

This slide showcases our complete dual-application framework in action. Given a single leaf image, the system provides:

**Application 1** – Disease prediction with confidence scores and top-3 predictions.

**Application 2** – Texture parameter extraction including roughness, anisotropy, complexity, and vein density. The system assigns the leaf to a texture cluster and compares it with typical cluster statistics.

The outcome is powerful: the system provides both biological and engineering interpretations from the same image. This demonstrates the power of a unified multi-domain pipeline, maximizing the value extracted from plant imagery.

---

## [Slide 18: Q&A] (5 seconds)

I'd now be happy to take any questions you might have.

---

## [Slide 19: Thank You] (10 seconds)

Thank you for your attention. If you're interested in exploring the code and data, the complete project is available on GitHub at the link shown. I look forward to your questions and feedback.

---

**TOTAL ESTIMATED TIME: ~8 minutes 15 seconds**

## Timing Breakdown:
- Slide 1: 15s
- Slide 2: 30s
- Slide 3: 30s
- Slide 4: 25s
- Slide 5: 30s
- Slide 6: 15s
- Slide 7: 35s
- Slide 8: 20s
- Slide 9: 20s
- Slide 10: 25s
- Slide 11: 20s
- Slide 12-13: 20s
- Slide 14: 45s (detailed cluster breakdown)
- Slide 15: 40s (corrected data emphasized)
- Slide 16: 25s
- Slide 17: 30s
- Slide 18: 5s
- Slide 19: 10s

**Total: 495 seconds ≈ 8 min 15 sec**

---

## Delivery Tips:

1. **Pacing**: Maintain steady, clear speech at approximately 130-150 words per minute
2. **Emphasis**: Stress key numbers (99.4% accuracy, 1351D → 50D, k=5)
3. **Transitions**: Use natural transitions like "Moving to...", "Now let's examine...", "This brings us to..."
4. **Eye Contact**: Look at audience during key points, glance at slides for data
5. **Gesture**: Point to visualizations when discussing clusters or graphs
6. **Pause**: Brief pauses after important statistics help audience absorb information
7. **Energy**: Maintain enthusiasm, especially when explaining the dual-application innovation
8. **Slide 15**: Emphasize the corrected data with confidence – this shows scientific rigor
