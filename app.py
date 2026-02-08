import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from fpdf import FPDF
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Institutional Portfolio Optimizer", layout="wide")

# --- 1. RÉCUPÉRATION DES DONNÉES RÉELLES (yfinance) ---
@st.cache_data
def get_market_data():
    tickers = {
        "Actions US": "VTI", 
        "Actions Int.": "VEU", 
        "Obligations": "AGG", 
        "Private Equity Proxy": "IWM", 
        "Immobilier": "VNQ",
        "RFR (T-Bill)": "BIL"
    }
    data = yf.download(list(tickers.values()), period="10y", interval="1mo")['Adj Close']
    returns = data.pct_change().dropna()
    returns.columns = [list(tickers.keys())[list(tickers.values()).index(col)] for col in returns.columns]
    return returns

# --- 2. LOGIQUE DE DÉLISSAGE (Geltner Model) ---
def desmooth_cov(cov, alpha, assets_to_desmooth=['Private Equity Proxy', 'Immobilier']):
    """Ajuste la covariance en cherchant les actifs par leur nom."""
    adj_cov = cov.copy()
    adj_factor = 1 / alpha
    for asset in assets_to_desmooth:
        if asset in adj_cov.index:
            adj_cov.loc[asset, :] *= adj_factor
            adj_cov.loc[:, asset] *= adj_factor
    return adj_cov

# --- 3. MOTEUR D'OPTIMISATION (CVXPY) ---
def optimize_portfolio(returns, cov, lev_limit, target_ret, borrow_cost, max_single, max_illiquid):
    n = len(returns)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov.values)
    
   # On identifie les positions dynamiquement
    idx_pe = list(returns.index).index('Private Equity Proxy')
    idx_re = list(returns.index).index('Immobilier')
    
    constraints = [
        cp.sum(w) <= lev_limit,
        cp.sum(w) >= 1.0,
        w >= 0,
        w <= max_single,
        w[idx_pe] + w[idx_re] <= max_illiquid, # Utilise les bons indices
        net_return >= target_ret
    ]
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- 4. GÉNÉRATION DE RAPPORT PDF ---
class InvestmentReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport d\'Optimisation Institutionnelle', 0, 1, 'C')
        self.ln(10)

def generate_pdf(weights, metrics, assets, loss):
    pdf = InvestmentReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", 0, 1)
    pdf.ln(5)
    for k, v in metrics.items():
        pdf.set_font("Arial", '', 11)
        pdf.cell(100, 10, f"{k}:", 0, 0)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"{v}", 0, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. INTERFACE UTILISATEUR (UI) ---
st.title("🏛️ Optimiseur de Fonds de Pension Multi-Actifs")

with st.sidebar:
    st.header("Configuration")
    lev_max = st.slider("Levier Max (Gross)", 1.0, 2.0, 1.2)
    target_r = st.slider("Cible Rendement (%)", 5.0, 12.0, 7.5) / 100
    alpha = st.slider("Alpha (Délissage PE/Immo)", 0.3, 1.0, 0.5)
    st.divider()
    st.header("Gouvernance")
    max_s = st.slider("Limite par actif (%)", 10, 50, 25) / 100
    max_i = st.slider("Limite Illiquides (%)", 10, 60, 40) / 100

# Exécution
hist_rets = get_market_data()
assets = hist_rets.columns
rfr = 0.035 # Taux sans risque fixe pour l'exemple
borrow_cost = rfr + 0.01

adj_cov = desmooth_cov(hist_rets.cov() * 12, alpha)
exp_rets = hist_rets.mean() * 12

w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, max_s, max_i)

if w_opt is not None:
    port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
    port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement Net", f"{port_ret:.2%}")
    col2.metric("Volatilité", f"{port_vol:.2%}")
    col3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

    # Graphiques
    c1, c2 = st.columns(2)
    with c1:
        fig_cap = px.pie(values=w_opt, names=assets, title="Allocation du Capital", hole=0.4)
        st.plotly_chart(fig_cap)
    with c2:
        mctr = (adj_cov @ w_opt) / port_vol
        tctr = (w_opt * mctr) / port_vol
        fig_risk = px.pie(values=tctr, names=assets, title="Contribution au Risque", hole=0.4)
        st.plotly_chart(fig_risk)

    # Backtest rapide
    st.divider()
    st.subheader("📊 Performance Historique du Portefeuille Optimisé")
    cum_rets = (1 + hist_rets.dot(w_opt)).cumprod() * 100
    st.line_chart(cum_rets)
    
    # Export PDF
    pdf_data = generate_pdf(w_opt, {"Ret": f"{port_ret:.2%}", "Vol": f"{port_vol:.2%}"}, assets, 0)
    st.download_button("📥 Télécharger Rapport PDF", data=pdf_data, file_name="rapport.pdf")
else:
    st.error("Impossible de trouver une solution avec ces contraintes. Augmentez le levier ou baissez la cible.")
