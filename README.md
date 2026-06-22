# Credit Risk Modeling: Champion vs Challengers

Binary classification project predicting loan default on 32,581 observations. Compares a logistic regression champion against three challengers (Random Forest, XGBoost, MLP), with threshold optimization via F2-score to reflect the asymmetric cost structure of credit risk.

**Course:** Data Science Software, M2 IRFA, Université Paris 1  
**Supervisor:** Bertrand Hassani  
**Authors:** Illian Hashatel, Reda Allab, Issa Ali Adoum

---

## Results

| Model | ROC-AUC | F2-score | Default recall |
|---|---|---|---|
| Logistic Regression (L1) | 0.857 | 0.708 | 0.578 |
| Random Forest | 0.935 | 0.793 | 0.728 |
| XGBoost | **0.950** | **0.826** | **0.783** |
| MLP | 0.906 | 0.744 | 0.729 |

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Place `credit_risk_dataset.csv` in the same directory as the notebook and run `credit_risk_modeling_en.ipynb` top to bottom.

The dataset is the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle.
