# 💎 Diamond Price Prediction & Market Segmentation  
### **End-to-End Machine Learning + ANN + Clustering Project**

##  Project Description
This project builds a complete end-to-end Machine Learning pipeline to predict **diamond prices** based on multiple physical and categorical features. The workflow covers **data cleaning, transformation, EDA, feature engineering, regression modeling, ANN model building, and clustering for market segmentation**. Multiple ML algorithms are compared to identify the most accurate and stable predictive model. The project also includes K-Means clustering to segment diamonds into meaningful groups based on their characteristics, helping businesses understand pricing tiers and customer patterns.

---

## **Project Workflow**

### **1️ Data Collection**
- Loaded the diamond dataset.
- Explored dataset structure, shape, and feature types.

### **2️ Data Understanding**
- Studied key attributes: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`, and `price`.

### **3️ Data Cleaning & Processing**
- Removed invalid or zero values in dimensions.
- Outlier detection and removal.
- Treated skewness using appropriate transformations.
- Scaled numerical features where required.

### **4️ Exploratory Data Analysis (EDA)**
- Distribution plots for numerical features.
- Boxplots to inspect outliers.
- Correlation heatmap.
- Price comparison based on cut, color, and clarity.
- Pairplots for relationship inspection.

### **5️ Feature Engineering**
- Ordinal Encoding for:
  - Cut
  - Color
  - Clarity
- Feature importance analysis using tree-based models.

---

## **Machine Learning Models Implemented**
- Linear Regression  
- Lasso Regression  
- Ridge Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- **Artificial Neural Network (ANN)** using TensorFlow/Keras

###  **Model Evaluation Metrics**
- R² Score  
- MAE  
- MSE  
- RMSE  

A comparison table is created to identify the best-performing model.

---

##  **Clustering – Market Segmentation**
- Implemented **K-Means** to group diamonds into meaningful segments.
- Helps analyze pricing tiers and identify diamond categories.
- Visualizations included for cluster separation.

---

##  **Technologies Used**
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- TensorFlow / Keras  
- XGBoost  

---

## 📁 **Project Structure**
