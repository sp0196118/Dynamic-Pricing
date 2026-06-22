"""
Streamlit Dashboard — Dynamic Pricing Engine
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")
st.title("💰 Dynamic Pricing Engine for Retail")
st.markdown("**DS Layer**: Demand elasticity model &nbsp;|&nbsp; **OR Layer**: Nonlinear profit maximisation")

np.random.seed(42)

PRODUCTS = {
    "Wireless Earbuds":  {"base_price": 1499, "base_demand": 200, "elasticity": -1.8, "cost": 600,  "stock": 500},
    "Phone Case":        {"base_price": 299,  "base_demand": 500, "elasticity": -2.5, "cost": 60,   "stock": 2000},
    "USB-C Hub":         {"base_price": 899,  "base_demand": 150, "elasticity": -1.3, "cost": 350,  "stock": 300},
    "Smart Watch Strap": {"base_price": 599,  "base_demand": 300, "elasticity": -2.0, "cost": 120,  "stock": 800},
}

@st.cache_data
def train_models():
    models = {}
    for product, cfg in PRODUCTS.items():
        bp, bd, e = cfg["base_price"], cfg["base_demand"], cfg["elasticity"]
        n = 500
        prices = np.random.uniform(bp*0.5, bp*1.8, n)
        dow = np.random.randint(0, 7, n)
        is_weekend = (dow >= 5).astype(float)
        comp_price = prices * np.random.uniform(0.8, 1.2, n)
        demand = (bd*(prices/bp)**e*(1+0.15*is_weekend)*(1+0.1*(comp_price>prices))
                  + np.random.normal(0, bd*0.05, n)).clip(0)
        X = pd.DataFrame({"price": prices, "price_ratio": prices/bp,
                          "is_weekend": is_weekend, "competitor_price": comp_price})
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X, demand)
        models[product] = model
    return models

with st.spinner("Training demand elasticity models..."):
    models = train_models()

with st.sidebar:
    st.header("⚙️ Settings")
    product = st.selectbox("Select Product", list(PRODUCTS.keys()))
    is_weekend = st.checkbox("Weekend pricing?", value=False)
    comp_ratio = st.slider("Competitor price ratio", 0.7, 1.3, 1.05, 0.05)
    cost_per_km = st.number_input("Override unit cost (₹)", value=PRODUCTS[product]["cost"])
    run = st.button("🚀 Optimise Price", type="primary")

cfg = PRODUCTS[product]
bp, cost, stock = cfg["base_price"], cfg["cost"], cfg["stock"]

if run:
    model = models[product]
    comp_p = bp * comp_ratio

    def predict_demand(p):
        X = pd.DataFrame([{"price": p, "price_ratio": p/bp,
                            "is_weekend": float(is_weekend), "competitor_price": comp_p}])
        return float(model.predict(X)[0])

    def neg_profit(p):
        d = min(predict_demand(p), stock)
        return -((p - cost_per_km) * d)

    res = minimize_scalar(neg_profit, bounds=(cost_per_km*1.05, bp*2.0), method="bounded")
    opt_price  = round(res.x, 2)
    opt_demand = min(predict_demand(opt_price), stock)
    opt_profit = (opt_price - cost_per_km) * opt_demand
    curr_profit = (bp - cost_per_km) * predict_demand(bp)
    uplift = (opt_profit - curr_profit) / max(curr_profit, 1) * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current Price", f"₹{bp:,}")
    c2.metric("Optimal Price", f"₹{opt_price:,}", f"{((opt_price/bp)-1)*100:+.1f}%")
    c3.metric("Expected Demand", f"{opt_demand:.0f} units")
    c4.metric("Profit Uplift", f"{uplift:+.1f}%")

    # Profit curve
    prices_range = np.linspace(cost_per_km*1.05, bp*2.0, 200)
    profits, demands = [], []
    for p in prices_range:
        d = min(predict_demand(p), stock)
        profits.append((p - cost_per_km) * d)
        demands.append(d)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_range, y=profits, name="Profit", line=dict(color="#5364FF", width=2)))
    fig.add_vline(x=opt_price, line_dash="dash", line_color="green", annotation_text=f"Optimal ₹{opt_price:,}")
    fig.add_vline(x=bp, line_dash="dot", line_color="gray", annotation_text=f"Current ₹{bp:,}")
    fig.update_layout(title=f"Profit Curve — {product}", xaxis_title="Price (₹)",
                      yaxis_title="Profit (₹)", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Demand curve
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prices_range, y=demands, name="Demand", line=dict(color="#E8503A")))
    fig2.add_vline(x=opt_price, line_dash="dash", line_color="green")
    fig2.update_layout(title="Demand Curve", xaxis_title="Price (₹)", yaxis_title="Units Demanded",
                       template="plotly_white", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    st.success(f"✅ Recommended price: **₹{opt_price:,}** (vs current ₹{bp:,}) — Profit uplift: **{uplift:+.1f}%**")
else:
    st.info("👈 Select a product and click **Optimise Price**")
