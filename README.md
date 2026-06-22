# 💰 Dynamic Pricing Engine for Retail

> **DS Layer**: Demand elasticity model (Gradient Boosting) &nbsp;|&nbsp; **OR Layer**: Nonlinear profit maximisation under margin + stock constraints

## 🎯 Problem Statement
What is the revenue-maximising price for each product, given that demand is elastic (changes with price), and subject to constraints on minimum margin and available stock?

## 🏗️ Architecture
```
Historical Price-Demand Data
        │
        ▼
  Demand Elasticity Model      ← DS Layer
  (GradientBoosting: features = price, ratio, weekend, competitor)
        │
        ▼
  Nonlinear Profit Maximisation  ← OR Layer
  Max: (price - cost) × demand_hat(price)
  Subject to: margin ≥ 5%, price ≤ 2× base, demand ≤ stock
        │
        ▼
  Profit curves + optimal price table
```

## 📦 Tech Stack
| Layer | Tool |
|-------|------|
| DS | `scikit-learn` GradientBoostingRegressor |
| OR | `scipy.optimize.minimize_scalar` (bounded) |
| Visualisation | `plotly`, `streamlit` |
| API | `FastAPI` (optional deploy) |

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python dynamic_pricing.py
streamlit run app.py
```

## 📊 Sample Output
```
Product              Current Price  Optimal Price  Price Change  Profit Uplift
Wireless Earbuds     1499           1389           -7.3%         +14.2%
Phone Case           299            249            -16.7%        +22.1%
USB-C Hub            899            1089           +21.1%        +8.7%
```

## 📁 Files
```
├── dynamic_pricing.py   # Full pipeline
├── app.py               # Streamlit interactive dashboard with sliders
├── requirements.txt
└── README.md
```
