import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fpdf import FPDF
import datetime

st.set_page_config(page_title="Optimiseur Pension Canada", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS (Perspective CAD / Non-Hedged) ---
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

@st.cache_data
def get_market_data(tickers):
    tickers_list = list(tickers.values())
    raw_data = yf.download(tickers_list, period="10y", interval="1mo")
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
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
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    asset_names = list(returns_series.index)
    ill_idx = [i for i, name in enumerate(asset_names) if name in illiquid_list]
    
    constraints = [
        cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0,
        w >= 0, w <= max_single,
        cp.sum(w[ill_idx]) <= max_illiquid,
        net_return >= target_ret
    ]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel Canadien")
st.markdown("Optimisation multi-actifs avec levier et gestion du risque de change (Unhedged).")

with st.sidebar:
    st.header("⚙️ Paramètres Stratégiques")
    lev_max = st.slider("Levier Maximum (Gross)", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage des alternatifs)", 0.3, 1.0, 0.5)
    
    st.header("⚖️ Gouvernance")
    max_s = st.slider("Max par actif (%)", 5,
