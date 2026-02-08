import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Équité Privée & Overlay", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS ---
# Ajout des Actions Privées via un Proxy (ex: PSP - Private Equity ETF)
TICKERS_DICT = {
    "Actions US (Unhedged)": "VFV.TO",
    "Actions Mondiales (Unhedged)": "VXC.TO",
    "Actions Privées (PE)": "PSP",
    "Marchés Émergents (Actions)": "VEE.TO",
    "Dette Marchés Émergents": "VWOB",
    "Infrastructures": "ZGI.TO",
    "Immobilier Listé": "VRE.TO",
    "Matières Premières": "DBC",
    "Petites Caps US (Unhedged)": "XSU.TO",
    "Obligations Can": "VAB.TO",
    "Dette Privée (Proxy)": "XHY.TO",
    "Hypothèques Comm. (Proxy)": "XCB.TO",
    "Cash (RFR)": "PSA.TO"
}

# Mise à jour des frais institutionnels par défaut
DEFAULT_MER = {
    "VFV.TO": 0.0009, "VXC.TO": 0.0021, "PSP": 0.0150, "VEE.TO": 0.0024, 
    "VWOB": 0.0020, "ZGI.TO": 0.0061, "VRE.TO": 0.0039, "DBC": 0.0085, 
    "XSU.TO": 0.0033, "VAB.TO": 0.0009, "XHY.TO": 0.0061, "XCB.TO": 0.0018, "PSA.TO": 0.0015
}

# Liste étendue des actifs illiquides soumis au délissage
ILLIQUID_ASSETS = ["Actions Privées (PE)", "Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    keys = list(tickers.keys())
    raw = yf.download([tickers[k] for k in keys], period="10y", interval="1mo")
    df = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df[keys].pct_change().dropna()

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    S = (cov_matrix.values + cov_matrix.values.T) / 2 + np.eye(n) * 1e-6
    risk = cp.quad_form(w, cp.psd_wrap(S))
    
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret, w >= 0]
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in ILLIQUID_ASSETS]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver)
            if w.value is not None: break
        except: continue
    return w.value if w.value is not None else None

# --- 2. INTERFACE SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Gouvernance CIO")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.5, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 12.0, 6.5) / 100
    
    st.subheader("🛡️ Risk Overlay")
    vol_target = st.slider("Cible Volatilité Max (%)", 4.0, 15.0, 9.0) / 100
    enable_deleveraging = st.checkbox("Activer De-leveraging Auto", value=True)
    
    alpha = st.slider("Alpha (Délissage PE/Privé)", 0.2, 1.0, 0.4)
    max_i = st.slider("Max Alternatifs Cumulés (%)", 10, 80, 50) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Méthode CMA :", ["Historique", "Manuel"])
    
    user_rets, user_vols = {}, {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            user_rets[asset] = st.sidebar.number_input(f"Rend. {asset} %", 8.0) / 100
            user_vols[asset] = st.sidebar.number_input(f"Vol. {asset} %", 15.0) / 100

# --- 3. ONGLETS ---
tab_opt, tab_risk, tab_hist, tab_lex = st.tabs(["📊 Optimisation", "📈 Corrélations", "🕒 Backtest", "🔍 Lexique"])

try:
    data = get_market_data(TICKERS_DICT)
    
    with tab_lex:
        st.header("📖 Glossaire Expert & Frais")
        st.markdown("""
        * **Actions Privées (PE) :** Investissements hors marchés publics. Prime d'illiquidité élevée mais risque de valorisation "lissée".
        * **De-leveraging :** Mécanisme de réduction de l'exposition globale pour maintenir le risque sous la barre des **{:.1%}** de volatilité.
        """.format(vol_target*100))
        applied_fees = {a: st.number_input(f"Frais {a} %", DEFAULT_MER[TICKERS_DICT[a]]*100, step=0.01, key=f"lex_{a}")/100 for a in TICKERS_DICT.keys()}

    with tab_opt:
        asset_bounds = {}
        cols = st.columns(4)
        for i, asset in enumerate(TICKERS_DICT.keys()):
            with cols[i % 4]:
                b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"m_{asset}")/100
                d_max = 0 if "Cash" in asset else 50
                b_max = st.number_input(f"Max {asset} %", 0, 100, d_max, key=f"M_{asset}")/100
                asset_bounds[asset] = (b_min, b_max)

        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        
        # CMA & Covariance
        if mode_cma == "Manuel":
            exp_raw = pd.Series(user_rets)
            v_diag = np.diag([user_vols[a] for a in TICKERS_DICT.keys()])
            cov_base = pd.DataFrame(v_diag @ data.corr().values @ v_diag, index=data.columns, columns=data.columns)
        else:
            exp_raw = data.mean()*12
            cov_base = data.cov()*12

        exp_rets = exp_raw - pd.Series(applied_fees)
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        # Optimisation
        w_opt = optimize_portfolio(exp_rets, cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i)
        
        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            p_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost)
            
            # Overlay
            status = "Nominal"
            if enable_deleveraging and p_vol > vol_target:
                w_opt = w_opt * (vol_target / p_vol)
                p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
                p_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost if np.sum(w_opt) > 1 else 0)
                status = "DE-LEVERAGING"

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exp. Net Return", f"{p_ret:.2%}")
            m2.metric("Portfolio Vol", f"{p_vol:.2%}")
            m3.metric("Leverage", f"{np.sum(w_opt):.2f}x")
            m4.metric("Risk Status", status)

            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Mix Actifs"), use_container_width=True)
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2 if p_vol > 0 else 1)
            c2.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Contribution au Risque (%)"), use_container_width=True)

    with tab_risk:
        fig_corr = px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=800)
        st.plotly_chart(fig_corr, use_container_width=True)

except Exception as e: st.error(f"Erreur : {e}")
