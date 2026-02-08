import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

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

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_illiquid, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    asset_names = list(returns_series.index)
    
    # Contraintes de base
    constraints = [
        cp.sum(w) <= lev_limit, 
        cp.sum(w) >= 1.0,
        net_return >= target_ret
    ]
    
    # Application des bornes par actif (Min/Max)
    for i, name in enumerate(asset_names):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    
    # Contrainte globale sur les actifs illiquides
    ill_idx = [i for i, name in enumerate(asset_names) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_illiquid)
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")

with st.sidebar:
    st.header("⚙️ Paramètres")
    lev_max = st.slider("Levier Brut Maximum", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible de Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    
    st.header("⚖️ Limites Alternatifs")
    max_i = st.slider("Plafond Alternatifs Globaux (%)", 10, 80, 45) / 100

    st.header("💳 Coût du Financement")
    spread_bps = st.number_input("Spread sur levier (bps)", value=120, step=10)

    st.header("🔮 Anticipations (CMA)")
    mode_cma = st.radio("Source :", ["Historique", "Manuel"])
    
    user_returns = {}
    user_vols = {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            user_returns[asset] = st.number_input(f"Rend. {asset} %", value=7.0, step=0.5) / 100
            user_vols[asset] = st.number_input(f"Vol. {asset} %", value=12.0, step=0.5) / 100

# --- NOUVELLE SECTION : BORNES PAR ACTIF ---
st.header("📊 Politique de Placement (Bornes Min/Max)")
st.info("Définissez les poids minimum et maximum autorisés pour chaque classe d'actif.")
asset_bounds = {}
cols = st.columns(4)
for i, asset in enumerate(TICKERS_DICT.keys()):
    with cols[i % 4]:
        st.subheader(asset)
        b_min = st.number_input(f"Min {asset} (%)", value=0, min_value=0, max_value=100, key=f"min_{asset}") / 100
        b_max = st.number_input(f"Max {asset} (%)", value=25, min_value=0, max_value=100, key=f"max_{asset}") / 100
        asset_bounds[asset] = (b_min, b_max)

try:
    hist_rets = get_market_data(TICKERS_DICT)
    corr_matrix = hist_rets.corr()
    rfr = (1 + hist_rets["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + (spread_bps / 10000)

    if mode_cma == "Manuel":
        exp_rets = pd.Series(user_returns)
        vols_diag = np.diag([user_vols[asset] for asset in hist_rets.columns])
        manual_cov = vols_diag @ corr_matrix.values @ vols_diag
        adj_cov_base = pd.DataFrame(manual_cov, index=hist_rets.columns, columns=hist_rets.columns)
    else:
        exp_rets = hist_rets.mean() * 12
        adj_cov_base = hist_rets.cov() * 12

    adj_cov = desmooth_cov(adj_cov_base, alpha, ILLIQUID_ASSETS)

    w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

    if w_opt is not None:
        port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
        port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
        
        st.divider()
        st.subheader("🎯 Résultat de l'Allocation")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rendement Net", f"{port_ret:.2%}")
        m2.metric("Volatilité Ajustée", f"{port_vol:.2%}")
        m3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

        # Frontière Efficiente
        st.divider()
        st.subheader("📈 Frontière Efficiente")
        target_range = np.linspace(max(0.02, exp_rets.min()), min(0.15, exp_rets.max()*lev_max), 10)
        f_vols, f_rets = [], []
        for r in target_range:
            w_tmp = optimize_portfolio(exp_rets, adj_cov, lev_max, r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)
            if w_tmp is not None:
                f_rets.append(w_tmp @ exp_rets - (np.sum(w_tmp)-1)*borrow_cost)
                f_vols.append(np.sqrt(w_tmp.T @ adj_cov @ w_tmp))
        
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=f_vols, y=f_rets, mode='lines+markers', name='Frontière', line=dict(color='gold')))
        fig_ef.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Sélection', marker=dict(color='red', size=12)))
        st.plotly_chart(fig_ef, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.pie(values=w_opt, names=hist_rets.columns, title="Répartition du Capital", hole=0.4), use_container_width=True)
        with col_b:
            risk_contrib = (w_opt * (adj_cov @ w_opt)) / (port_vol**2)
            st.plotly_chart(px.pie(values=risk_contrib, names=hist_rets.columns, title="Répartition du Risque", hole=0.4), use_container_width=True)

        st.divider()
        st.subheader("🏁 Backtest vs 60/40")
        bench_rets = (hist_rets["Actions Mondiales (Unhedged)"] * 0.60) + (hist_rets["Obligations Can"] * 0.40)
        port_rets_ts = hist_rets.dot(w_opt)
        comp_df = pd.DataFrame({"Optimisé": (1 + port_rets_ts).cumprod() * 100, "60/40": (1 + bench_rets).cumprod() * 100}, index=hist_rets.index)
        st.line_chart(comp_df)
    else:
        st.error("⚠️ Impossible de trouver une solution respectant toutes vos contraintes. Essayez d'élargir vos bornes Min/Max.")

except Exception as e:
    st.error(f"Erreur technique : {e}")
