"""
Dynamic Pricing Engine for Retail
====================================
DS Layer  : Demand elasticity model — predict demand given price (XGBoost)
OR Layer  : Nonlinear profit maximisation (scipy.optimize) under margin + stock constraints
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC PRICE-DEMAND DATA
#    (replace with real POS / e-commerce export)
# ─────────────────────────────────────────────
PRODUCTS = {
    "Wireless Earbuds":  {"base_price": 1499, "base_demand": 200, "elasticity": -1.8, "cost": 600,  "stock": 500},
    "Phone Case":        {"base_price": 299,  "base_demand": 500, "elasticity": -2.5, "cost": 60,   "stock": 2000},
    "USB-C Hub":         {"base_price": 899,  "base_demand": 150, "elasticity": -1.3, "cost": 350,  "stock": 300},
    "Smart Watch Strap": {"base_price": 599,  "base_demand": 300, "elasticity": -2.0, "cost": 120,  "stock": 800},
}

def generate_price_demand(product_name, config, n=500):
    """Simulate historical price experiments with noise."""
    bp, bd, e = config["base_price"], config["base_demand"], config["elasticity"]
    prices = np.random.uniform(bp * 0.5, bp * 1.8, n)
    # Demand = base_demand * (price / base_price)^elasticity + noise + day/season effects
    day_of_week = np.random.randint(0, 7, n)
    is_weekend  = (day_of_week >= 5).astype(float)
    competitor_price = prices * np.random.uniform(0.8, 1.2, n)
    demand = (
        bd * (prices / bp) ** e
        * (1 + 0.15 * is_weekend)
        * (1 + 0.1 * (competitor_price > prices))
        + np.random.normal(0, bd * 0.05, n)
    ).clip(0)

    return pd.DataFrame({
        "product": product_name,
        "price": prices,
        "demand": demand,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "competitor_price": competitor_price,
        "price_ratio": prices / bp,
    })

all_data = pd.concat([generate_price_demand(k, v) for k, v in PRODUCTS.items()], ignore_index=True)
print(f"[DATA] {len(all_data)} price-demand observations for {len(PRODUCTS)} products")

# ─────────────────────────────────────────────
# 2. DS LAYER — DEMAND ELASTICITY MODEL
# ─────────────────────────────────────────────
FEATURES = ["price", "price_ratio", "is_weekend", "competitor_price"]

models = {}
scores = {}
for product in PRODUCTS:
    df_p = all_data[all_data.product == product]
    X = df_p[FEATURES]; y = df_p["demand"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    mape = mean_absolute_percentage_error(y_te, model.predict(X_te))
    models[product] = model
    scores[product] = mape
    print(f"[MODEL] {product:<25} MAPE = {mape:.2%}")

# ─────────────────────────────────────────────
# 3. OR LAYER — PROFIT MAXIMISATION
#    For each product:
#    Maximise  profit(p) = (p - cost) * demand_hat(p)
#    Subject to:
#      p >= cost * 1.05          (minimum 5% margin)
#      p <= base_price * 2.0     (maximum price ceiling)
#      demand_hat(p) <= stock    (stock constraint)
# ─────────────────────────────────────────────
results = []

for product, config in PRODUCTS.items():
    model   = models[product]
    cost    = config["cost"]
    stock   = config["stock"]
    bp      = config["base_price"]
    comp_p  = bp * 1.05   # assume competitor slightly above base

    def predict_demand(p):
        X = pd.DataFrame([{
            "price": p,
            "price_ratio": p / bp,
            "is_weekend": 0.3,          # average weekend probability
            "competitor_price": comp_p
        }])
        return float(model.predict(X)[0])

    def neg_profit(p):
        d = predict_demand(p)
        d = min(d, stock)               # stock constraint
        return -((p - cost) * d)        # negative because we minimise

    bounds = (cost * 1.05, bp * 2.0)
    res = minimize_scalar(neg_profit, bounds=bounds, method="bounded")

    opt_price  = round(res.x, 2)
    opt_demand = min(predict_demand(opt_price), stock)
    opt_profit = (opt_price - cost) * opt_demand
    curr_profit = (bp - cost) * predict_demand(bp)
    uplift = (opt_profit - curr_profit) / curr_profit * 100

    results.append({
        "Product":         product,
        "Current Price":   bp,
        "Optimal Price":   opt_price,
        "Price Change":    f"{((opt_price/bp)-1)*100:+.1f}%",
        "Expected Demand": round(opt_demand, 0),
        "Current Profit":  round(curr_profit, 0),
        "Optimal Profit":  round(opt_profit, 0),
        "Profit Uplift":   f"{uplift:+.1f}%",
        "Stock":           stock,
        "Cost":            cost,
    })

df_results = pd.DataFrame(results)
print("\n[OR] Optimal Pricing Results:")
print(df_results[["Product","Current Price","Optimal Price","Price Change","Profit Uplift"]].to_string(index=False))
df_results.to_csv("optimal_prices.csv", index=False)

# ─────────────────────────────────────────────
# 4. PROFIT CURVE VISUALISATION
# ─────────────────────────────────────────────
fig = go.Figure()
for product, config in PRODUCTS.items():
    model   = models[product]
    bp, cost, stock = config["base_price"], config["cost"], config["stock"]
    comp_p  = bp * 1.05
    prices  = np.linspace(cost * 1.05, bp * 2.0, 200)
    profits = []
    for p in prices:
        X = pd.DataFrame([{"price": p, "price_ratio": p/bp, "is_weekend": 0.3, "competitor_price": comp_p}])
        d = min(float(model.predict(X)[0]), stock)
        profits.append((p - cost) * d)
    fig.add_trace(go.Scatter(x=prices, y=profits, name=product, mode="lines"))

fig.update_layout(title="Profit Curves by Price Point", xaxis_title="Price (₹)", yaxis_title="Profit (₹)",
                  template="plotly_white", height=450)
fig.write_html("profit_curves.html")
print("[VIZ] Saved → profit_curves.html")
print("[DONE] Results → optimal_prices.csv")
