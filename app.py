import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Dashboard Strategique Pension", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS ---
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

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    raw_data = yf.download(list(tickers.values()), period="10y", interval="1mo")
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
st.title("🏛️ Analyse et Optimisation de Portefeuille Institutionnel")

with st.sidebar:
    st.header("Levier et Rendement")
    lev_max = st.slider("Levier Maximum", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    st.header("Limites de Concentration")
    max_s = st.slider("Max par actif (%)", 5, 40, 20) / 100
    max_i = st.slider("Max Illiquides (%)", 10, 80, 45) / 100

try:
    hist_rets = get_market_data(TICKERS_DICT)
    exp_rets = hist_rets.mean() * 12
    rfr = (1 + hist_rets["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + 0.012
    adj_cov = desmooth_cov(hist_rets.cov() * 12, alpha, ILLIQUID_ASSETS)

    # --- NOUVELLE SECTION : VISUALISATION DES HYPOTHÈSES ---
    st.divider()
    col_data1, col_data2 = st.columns(2)
    
    with col_data1:
        with st.expander("🔍 Voir les statistiques par classe d'actif"):
            stats_df = pd.DataFrame({
                "Rendement (%)": exp_rets * 100,
                "Volatilité Brute (%)": (hist_rets.std() * np.sqrt(12)) * 100,
                "Sharpe": (exp_rets - rfr) / (hist_rets.std() * np.sqrt(12))
            })
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

    with col_data2:
        with st.expander("📈 Voir la matrice de corrélation"):
            corr = hist_rets.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
    st.divider()

    # --- OPTIMISATION ---
    w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, max_s, max_i, ILLIQUID_ASSETS)

    if w_opt is not None:
        port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
        port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendement Net", f"{port_ret:.2%}")
        c2.metric("Volatilité Ajustée", f"{port_vol:.2%}")
        c3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_cap = px.pie(values=w_opt, names=hist_rets.columns, title="Répartition du Capital", hole=0.4)
            st.plotly_chart(fig_cap, use_container_width=True)
        with col_b:
            mctr = (adj_cov @ w_opt) / port_vol
            tctr = (w_opt * mctr) / port_vol
            fig_risk = px.pie(values=tctr, names=hist_rets.columns, title="Répartition du Risque", hole=0.4)
            st.plotly_chart(fig_risk, use_container_width=True)
            
        st.subheader("📊 Croissance Historique du Portefeuille")
        st.line_chart((1 + hist_rets.dot(w_opt)).cumprod() * 100)
    else:
        st.warning("⚠️ Contraintes trop strictes. Solution impossible.")

except Exception as e:
    st.error(f"Erreur technique : {e}")
