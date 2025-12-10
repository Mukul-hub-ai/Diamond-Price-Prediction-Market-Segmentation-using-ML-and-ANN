# Diamond Price Prediction (Machine Learning + ANN + Clustering)

This project predicts the price of diamonds based on features like carat, cut, color, clarity, depth, table and dimensions (x, y, z). Along with price prediction, K-Means clustering is used to identify diamond groups based on quality and characteristics.

---

## 1. Objective
- To build a machine learning system that predicts diamond prices accurately.
- To understand which features influence diamond prices the most.
- To segment diamonds into groups (clusters) for market insights.

---

## 2. Dataset Description
- Total features: Numerical + Categorical  
- Target variable: **price**
- Major features include:
  - carat  
  - cut  
  - color  
  - clarity  
  - depth, table  
  - x, y, z dimensions

---

## 3. Steps Performed

### ✔ Data Cleaning
- Removed rows with zero or invalid x, y, z values  
- Checked missing values  
- Treated major outliers  

### ✔ Data Transformation
- Categorical encoding (cut, color, clarity)
- Standardization for numerical features
- Skewness correction (log/box-cox)

### ✔ Exploratory Data Analysis
- Distribution analysis  
- Boxplots  
- Correlation heatmap  
- Price comparison across categories  

### ✔ Feature Engineering
- Ordinal encoding  
- Feature importance using RandomForest/XGBoost  

---

## 4. Machine Learning Models Used
- Linear Regression  
- Ridge / Lasso Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- **Artificial Neural Network (ANN)**

### Evaluation Metrics
- R² Score  
- MAE  
- RMSE  

---

## 5. Clustering (K-Means)
- Applied K-Means clustering  
- Elbow Method used to select optimal K  
- Cluster visualization for quality segmentation  

---

## 6. Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- TensorFlow/Keras  
- XGBoost  
- Jupyter Notebook  

**Mukul**  
