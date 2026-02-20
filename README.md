# 🏠 Real Estate Price Prediction System

**Production ML Pipeline + Interactive Streamlit Predictor**  
*End-to-End Analysis → Deployment on 2.2M+ US Real Estate Listings*

Predict house prices in **3 clicks** using bedrooms, bathrooms, and house size. Built with scikit-learn LinearRegression on Realtor.com dataset (**1.3M cleaned rows**). **MAE: $353K** on test set.


---

## 🎯 Business Value

| Stakeholder | Value Delivered |
|-------------|-----------------|
| **Real Estate Agents** | Instant comps (4 bed/3 bath/2500sqft → ~$650K) |
| **Buyers/Sellers** | Data-driven pricing across 50+ states |
| **Production** | joblib models + <100ms Streamlit inference |

---

## 📊 Dataset Insights

**2,226,382 raw rows → 1,355,862 cleaned rows** (45% data quality improvement)

| Metric | Raw | Cleaned | Insight |
|--------|-----|---------|---------|
| **Rows** | 2.22M | **1.36M** | Dropped NaNs/duplicates |
| **States** | 50+ | 50+ | CA/HI/DC highest prices |
| **Price Range** | $0 - $2.1B | Median $325K | Outliers handled |
| **Key Correlations** | - | **bath (0.21), bed (0.12)** | Strongest price drivers |

### 🥇 Top 5 States by Avg Price

Virgin Islands: $1,947,522

Hawaii: $1,491,548

California: $1,107,572

DC: $1,209,649

Colorado: $952,486

---

## 🔬 Production Methodology

### 1. **Data Engineering (ETL Pipeline)**

Handled 1.9M NaNs → Strategic column pruning + dropna()
Removed 535 duplicates → 1,355,862 production-ready rows
Feature selection: bed/bath/house_size (corr >0.07)

### 2. **EDA & Insights**

- **State distribution**: CA/WA volume leaders
- **Correlation matrix**: bath (r=0.21) > bed > house_size
- **City outliers**: International listings hit $2.1B avg

### 3. **ML Pipeline**

```python
X = ["bed", "bath", "house_size"]  # 80/20 split
scaler = StandardScaler().fit_transform(X_train)  # joblib.dump
model = LinearRegression().fit(X_train, y_train)  # MAE $353K
```

Key Decisions:

Ignored status column for general pricing model
Linear baseline (strong for portfolio + interpretable)

🚀 Live Streamlit Demo

👉 Enter: 4 beds | 3 baths | 2500 sqft
✅ Predicts: $XXX,XXX (real-time)
🎈 Responsive UI + celebration animation
⚡ Scaler/model auto-loaded (<50ms inference)

Deploy: streamlit run app.py
<img width="1910" height="952" alt="Screenshot 2026-02-19 190441" src="https://github.com/user-attachments/assets/aabe678e-23b0-45dc-8240-ddb531717f16" />
<img width="1910" height="952" alt="Screenshot 2026-02-19 190441" src="https://github.com/user-attachments/assets/c1b79a8f-6c79-40da-9b8a-38ffbfa7211d" />
