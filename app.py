import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Optimiseur Pension Canada", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS (Focus Canada & Alternatifs) ---
TICKERS_DICT = {
    "Actions US (Hedged)": "VSP.TO",
    "Actions Mondiales": "VXC.TO",
    "Marchés Émergents": "VEE.TO",
    "Infrastructures": "ZGI.TO",
    "Immobilier Listé": "VRE.TO",
    "Matières Premières": "DBC",
    "Petites Caps US": "IWM",
    "Obligations Can": "VAB.TO",
    "Dette Privée (Proxy)": "XHY.TO",
    "Hypothèques Comm. (Proxy)": "XCB.TO",
    "Cash (RFR)": "PSA.TO"
}

# Liste des actifs considérés comme "illiquides" pour le délissage
ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    raw_data = yf.download(list(tickers.values()), period="10y", interval="1mo")
    # Extraction sécurisée du prix de clôture ajusté
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
    returns = data.pct_change().dropna()
    inv_tickers = {v: k for k, v in tickers.items()}
    return returns.rename(columns=inv_tickers)

# --- 2. LOGIQUE DE DÉLISSAGE ---
def desmooth_cov(cov, alpha, illiquid_list):
    adj_cov = cov.copy()
    adj_factor = 1 / alpha
    for asset in illiquid_list:
        if asset in adj_cov.index:
            adj_cov.loc[asset, :] *= adj_factor
            adj_cov.loc[:, asset] *= adj_factor
    return adj_cov

# --- 3. MOTEUR D'OPTIMISATION ---
def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, max_single, max_illiquid, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    
    asset_names = list(returns_series.index)
    illiquid_indices = [i for i, name in enumerate(asset_names) if name in illiquid_list]
    
    constraints = [
        cp.sum(w) <= lev_limit,
        cp.sum(w) >= 1.0,
        w >= 0,
        w <= max_single,
        cp.sum(w[illiquid_indices]) <= max_illiquid,
        net_return >= target_ret
    ]
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- 4. INTERFACE UTILISATEUR ---
st.title("🏛️ Portefeuille Multi-Actifs : Perspective Canadienne")

with st.sidebar:
    st.header("Paramètres Stratégiques")
    lev_max = st.slider("Levier Maximum", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement Annuel (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Facteur de Délissage (Illiquides)", 0.3, 1.0, 0.5)
    
    st.header("Contraintes de Gouvernance")
    max_s = st.slider("Limite par actif (%)", 5, 40, 20) / 100
    max_i = st.slider("Limite totale Illiquides (%)", 10, 80, 45) / 100

try:
    hist_rets = get_market_data(TICKERS_DICT)
    # Calcul des paramètres de marché
    exp_rets = hist_rets.mean() * 12
    # On utilise PSA.TO (Cash) comme taux sans risque actuel
    rfr = (1 + hist_rets["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + 0.012 # Spread institutionnel typique

    adj_cov = desmooth_cov(hist_rets.cov() * 12, alpha, ILLIQUID_ASSETS)

    w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, max_s, max_i, ILLIQUID_ASSETS)

    if w_opt is not None:
        port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
        port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
        
        st.subheader("📊 Analyse du Portefeuille Optimal")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendement Net Estimé", f"{port_ret:.2%}")
        c2.metric("Volatilité (Ajustée)", f"{port_vol:.2%}")
        c3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            # Correction de la parenthèse fermante ici
            fig_cap = px.pie(values=w_opt, names=hist_rets.columns, title="Allocation par Classe d'Actifs", hole=0.4)
            st.plotly_chart(fig_cap, use_container_width=True)
        with col_b:
            mctr = (adj_cov @ w_opt) / port_vol
            tctr = (w_opt * mctr) / port_vol
            # Correction de la parenthèse fermante ici
            fig_risk = px.pie(values=tctr, names=hist_rets.columns, title="Contribution au Risque (Volatilité)", hole=0.4)
            st.plotly_chart(fig_risk, use_container_width=True)
