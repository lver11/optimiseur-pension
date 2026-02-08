import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf
from fpdf import FPDF
import datetime

st.set_page_config(page_title="Institutional Portfolio Optimizer", layout="wide")

# --- 1. RÉCUPÉRATION DES DONNÉES ---
@st.cache_data
def get_market_data():
    # Dictionnaire des tickers pour identifier chaque classe
    tickers = {
        "Actions US": "VTI", 
        "Actions Int.": "VEU", 
        "Obligations": "AGG", 
        "Private Equity Proxy": "IWM", 
        "Immobilier": "VNQ",
        "RFR (T-Bill)": "BIL"
    }
    raw_data = yf.download(list(tickers.values()), period="10y", interval="1mo")['Adj Close']
    returns = raw_data.pct_change().dropna()
    
    # Inversion du dictionnaire pour renommer les colonnes selon les clés (ex: VTI -> Actions US)
    inv_tickers = {v: k for k, v in tickers.items()}
    returns = returns.rename(columns=inv_tickers)
    return returns

# --- 2. LOGIQUE DE DÉLISSAGE ---
def desmooth_cov(cov, alpha):
    adj_cov = cov.copy()
    adj_factor = 1 / alpha
    # On applique le délissage si l'actif est présent
    for asset in ["Private Equity Proxy", "Immobilier"]:
        if asset in adj_cov.index:
            adj_cov.loc[asset, :] *= adj_factor
            adj_cov.loc[:, asset] *= adj_factor
    return adj_cov

# --- 3. MOTEUR D'OPTIMISATION ---
def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, max_single, max_illiquid):
    n = len(returns_series)
    w = cp.Variable(n)
    
    # Calcul du rendement net (Rendement Brut - Coût du Levier)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    
    # Identification dynamique des indices pour les contraintes
    asset_names = list(returns_series.index)
    illiquid_indices = [i for i, name in enumerate(asset_names) if name in ["Private Equity Proxy", "Immobilier"]]
    
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

# --- 4. INTERFACE ---
st.title("🏛️ Optimiseur de Fonds de Pension")

with st.sidebar:
    st.header("Configuration")
    lev_max = st.slider("Levier Max (Gross)", 1.0, 2.0, 1.3)
    target_r = st.slider("Cible Rendement (%)", 4.0, 12.0, 7.0) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    st.divider()
    st.header("Gouvernance")
    max_s = st.slider("Limite par actif (%)", 10, 50, 25) / 100
    max_i = st.slider("Limite Illiquides (%)", 10, 80, 40) / 100

try:
    hist_rets = get_market_data()
    assets = hist_rets.columns
    rfr = 0.035
    borrow_cost = rfr + 0.01

    adj_cov = desmooth_cov(hist_rets.cov() * 12, alpha)
    exp_rets = hist_rets.mean() * 12

    w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, max_s, max_i)

    if w_opt is not None:
        # Calcul des métriques
        port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
        port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
        
        st.subheader("📊 Métriques Clés")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendement Net", f"{port_ret:.2%}")
        c2.metric("Volatilité (Délissée)", f"{port_vol:.2%}")
        c3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

        # Graphiques
        col_a, col_b = st.columns(2)
        with col_a:
            fig_cap = px.pie(values=w_opt, names=assets, title="Allocation du Capital", hole=0.4)
            st.plotly_chart(fig_cap, use_container_width=True)
        with col_b:
            mctr = (adj_cov @ w_opt) / port_vol
            tctr = (w_opt * mctr) / port_vol
            fig_risk = px.pie(values=tctr, names=assets, title="Contribution au Risque", hole=0.4)
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.warning("⚠️ Aucune solution trouvée. Essayez d'augmenter le levier ou de réduire la cible de rendement.")

except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
