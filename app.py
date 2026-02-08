import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import datetime

# Configuration de la page
st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS (Non-Hedged / Perspective CAD) ---
TICKERS_DICT = {
    "Actions US (Unhedged)": "VFV.TO",
    "Actions Mondiales (Unhedged)": "VXC.TO",
    "Marchés Émergents": "VEE.TO",
    "Infrastructures": "ZGI.TO",
    "Immobilier Listé": "VRE.TO",
    "Matières Premières": "DBC",
    "Petites Caps US (Unhedged)": "XSU.TO",
    "Obligations Can": "VAB.TO",
    "Dette Privée (Proxy)": "XHY.TO",
    "Hypothèques Comm. (Proxy)": "XCB.TO",
    "Cash (RFR)": "PSA.TO"
}

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

# --- 2. RÉCUPÉRATION ET TRAITEMENT DES DONNÉES ---
@st.cache_data
def get_market_data(tickers):
    tickers_list = list(tickers.values())
    raw_data = yf.download(tickers_list, period="10y", interval="1mo")
    
    # Gestion du format Multi-index de yfinance
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']
        
    returns = data.pct_change().dropna()
    inv_tickers = {v: k for k, v in tickers.items()}
    return returns.rename(columns=inv_tickers)

def desmooth_cov(cov, alpha, illiquid_list):
    adj_cov = cov.copy()
    adj_factor = 1 / alpha
    for asset in illiquid_list:
        if asset in adj_cov.index:
            adj_cov.loc[asset, :] *= adj_factor
            adj_cov.loc[:, asset] *= adj_factor
    return adj_cov

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, max_single, max_illiquid, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    
    # Coût du levier sur l'excès de 100%
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    
    asset_names = list(returns_series.index)
    ill_idx = [i for i, name in enumerate(asset_names) if name in illiquid_list]
    
    constraints = [
        cp.sum(w) <= lev_limit,
        cp.sum(w) >= 1.0,
        w >= 0,
        w <= max_single,
        cp.sum(w[ill_idx]) <= max_illiquid,
        net_return >= target_ret
    ]
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- 3. INTERFACE UTILISATEUR ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel Canadien")

with st.sidebar:
    st.header("⚙️ Paramètres de Gestion")
    lev_max = st.slider("Levier Brut Maximum", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible de Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Facteur de Délissage (Alpha)", 0.3, 1.0, 0.5)
    
    st.header("⚖️ Limites de Risque")
    max_s = st.slider("Max par actif (%)", 5, 40, 20) / 100
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100

try:
    hist_rets = get_market_data(TICKERS_DICT)
    exp_rets = hist_rets.mean() * 12
    rfr = (1 + hist_rets["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + 0.012
    adj_cov = desmooth_cov(hist_rets.cov() * 12, alpha, ILLIQUID_ASSETS)

    # --- AFFICHAGE DES HYPOTHÈSES ---
    st.divider()
    h1, h2 = st.columns(2)
    with h1:
        with st.expander("🔍 Hypothèses de Marché (CMA Historiques)"):
            stats = pd.DataFrame({
                "Rendement (%)": exp_rets * 100,
                "Volatilité (%)": (hist_rets.std() * np.sqrt(12)) * 100,
                "Sharpe": (exp_rets - rfr) / (hist_rets.std() * np.sqrt(12))
            })
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
    with h2:
        with st.expander("📈 Corrélations (Perspective CAD)"):
            fig_corr = px.imshow(hist_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- OPTIMISATION ---
    w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, max_s, max_i, ILLIQUID_ASSETS)

    if w_opt is not None:
        port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
        port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
        
        st.subheader("🎯 Allocation Optimale")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rendement Net Estimé", f"{port_ret:.2%}")
        m2.metric("Volatilité (Délissée)", f"{port_vol:.2%}")
        m3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.pie(values=w_opt, names=hist_rets.columns, title="Répartition du Capital", hole=0.4), use_container_width=True)
        with col_b:
            # Contribution au risque simplifiée
            risk_contrib = (w_opt * (adj_cov @ w_opt)) / (port_vol**2)
            st.plotly_chart(px.pie(values=risk_contrib, names=hist_rets.columns, title="Répartition du Risque", hole=0.4), use_container_width=True)

        # --- COMPARATEUR 60/40 & DRAWDOWN ---
        st.divider()
        st.subheader("🏁 Comparaison Stratégique : vs Benchmark 60/40")
        
        # Benchmark : 60% Monde Unhedged / 40% Oblig Can
        bench_rets = (hist_rets["Actions Mondiales (Unhedged)"] * 0.60) + (hist_rets["Obligations Can"] * 0.40)
        port_rets_ts = hist_rets.dot(w_opt)
        
        comp_df = pd.DataFrame({
            "Portefeuille Optimisé": (1 + port_rets_ts).cumprod() * 100,
            "Benchmark 60/40": (1 + bench_rets).cumprod() * 100
        }, index=hist_rets.index)
        
        st.line_chart(comp_df)

        # Drawdowns
        dd_opt = (comp_df["Portefeuille Optimisé"] / comp_df["Portefeuille Optimisé"].cummax()) - 1
        dd_bench = (comp_df["Benchmark 60/40"] / comp_df["Benchmark 60/40"].cummax()) - 1
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_opt.index, y=dd_opt*100, fill='tozeroy', name='Optimisé', line=dict(color='blue')))
        fig_dd.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench*100
