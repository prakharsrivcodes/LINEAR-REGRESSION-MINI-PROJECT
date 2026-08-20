
# Global Life Expectancy Prediction & Socio-Economic Analysis
An end-to-end Machine Learning and Data Analysis project predicting country-level Life Expectancy using the World Health Organization (WHO) dataset.

---

## 📌 Project Overview
Predicting life expectancy is critical for understanding national health outcomes and socio-economic development. This project evaluates global health, immunization, economic, and demographic metrics across **193 countries (2000–2015)** to build a regularized predictive regression model.

Rather than relying on simplified synthetic datasets, this project addresses real-world data engineering challenges:
- **High-density missing values** across multi-year country records.
- **Multicollinearity** among socio-economic indicators (e.g., infant vs. under-five mortality, thinness metrics).
- **Feature Selection** using $L_1$ Regularization (Lasso) to simplify model complexity without sacrificing predictive power.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `LinearRegression`, `Lasso`, `Ridge`, `train_test_split`)
- **Data Visualization:** `matplotlib`, `seaborn`

---

## 🔬 Key Pipeline Steps

### 1. Data Cleaning & Header Normalization
- Handled raw Kaggle CSV whitespace anomalies in column headers using string trimming (`str.strip()`).
- Dropped 10 records missing the target variable (`Life expectancy`) to prevent target leakage/bias during model evaluation.

### 2. Grouped Smart Imputation Strategy
- **Primary Imputation:** Applied country-level median imputation (`df.groupby('Country')`) for missing indicators like `GDP`, `Schooling`, `Alcohol`, and `Population`. This preserved country-specific socio-economic baselines rather than applying naive global averages.
- **Secondary Fallback:** Utilized global medians for countries missing entire multi-year metric records (e.g., South Sudan).

### 3. Categorical Encoding & Feature Scaling
- Encoded binary categorical indicator `Status` (`Developing` $
ightarrow$ `0`, `Developed` $
ightarrow$ `1`).
- Applied `StandardScaler` on input features ($X$) to standardize scale variances prior to running $L_1$/$L_2$ regularized models.

### 4. Regularization & Feature Selection
Evaluated Ordinary Least Squares (OLS) Linear Regression against $L_1$ (Lasso) and $L_2$ (Ridge) Regularization techniques.

---

## 📊 Model Performance & Findings

| Model | $R^2$ Score | RMSE (Years) | Key Feature Action |
| :--- | :---: | :---: | :--- |
| **Linear Regression (OLS)** | `0.8196` | `3.950` | Retains all 20 features (susceptible to multicollinearity). |
| **Lasso Regression ($ lpha=0.1$)** | `0.8098` | `4.055` | **Dropped 4 redundant features** (`Year`, `Hepatitis B`, `infant deaths`, `thinness 5-9 years`). |
| **Ridge Regression ($ lpha=1.0$)** | `0.8195` | `3.951` | Penalizes large coefficients smoothly while keeping all features. |

### 💡 Key Drivers Identified
- **Positive Predictors:** `Schooling` (+2.19 weight) and `Income composition of resources` (+1.21 weight) are the strongest positive drivers of life expectancy.
- **Negative Predictors:** `Adult Mortality` (-2.58 weight) and `HIV/AIDS` (-2.45 weight) represent the most severe negative impacts on national lifespan.
- **Multicollinearity Removal:** Lasso successfully identified collinear features (e.g., `infant deaths` was redundant alongside `under-five deaths`) and eliminated them by shrinking their coefficients to **exact 0**.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/life-expectancy-prediction.git
   cd life-expectancy-prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Execute the Notebook:**
   Open `linear_reg_project.ipynb` in Google Colab or Jupyter Notebook, upload `Life Expectancy Data.csv`, and run all cells.

