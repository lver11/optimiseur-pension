import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Contrôle Dynamique", layout="wide")

# --- 1. CONFIGURATION ---
TICKERS_DICT = {
    "Actions US (Unhedged)": "VFV.TO",
    "Actions Mondiales (Unhedged)": "VXC.TO",
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

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

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
    st.header("⚙️ Gouvernance & Risk Overlay")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    
    st.subheader("🛡️ Target Volatility Trigger")
    vol_target = st.slider("Cible Volatilité Max (%)", 5.0, 15.0, 8.5) / 100
    enable_deleveraging = st.checkbox("Activer De-leveraging Auto", value=True)
    
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("CMA Source :", ["Historique", "Manuel"])

# --- 3. TRAITEMENT ---
try:
    data = get_market_data(TICKERS_DICT)
    tab_opt, tab_risk, tab_lex = st.tabs(["📊 Optimisation", "📈 Analyse Risque", "🔍 Lexique"])

    with tab_lex:
        st.header("📖 Lexique Professionnel")
        st.markdown("""
        * **De-leveraging Auto :** Si la volatilité optimisée dépasse la cible, l'outil réduit le levier proportionnellement pour ramener le risque au niveau souhaité, quitte à sacrifier la cible de rendement.
        * **Target Volatility :** Méthodologie consistant à ajuster l'exposition au marché pour maintenir un niveau de risque constant.
        """)
        applied_fees = {a: st.number_input(f"Frais {a} %", 0.20, key=f"lex_f_{a}")/100 for a in TICKERS_DICT.keys()}

    with tab_opt:
        asset_bounds = {}
        cols = st.columns(4)
        for i, asset in enumerate(TICKERS_DICT.keys()):
            with cols[i % 4]:
                b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"min_{asset}")/100
                b_max = st.number_input(f"Max {asset} %", 0, 100, 40 if "Cash" not in asset else 0, key=f"max_{asset}")/100
                asset_bounds[asset] = (b_min, b_max)

        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        exp_raw = data.mean()*12
        cov_base = data.cov()*12
        
        # Ajustement Délissage
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        # 1. Optimisation Initiale
        w_opt = optimize_portfolio(exp_raw - pd.Series(applied_fees), cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i)
        
        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            p_ret = (w_opt @ (exp_raw - pd.Series(applied_fees))) - ((np.sum(w_opt)-1) * borrow_cost)
            
            # 2. Logic de De-leveraging
            deleveraging_status = "Inactif"
            if enable_deleveraging and p_vol > vol_target:
                deleveraging_status = "ACTIF"
                adjustment_factor = vol_target / p_vol
                # On réduit le levier en augmentant le Cash ou en réduisant les positions proportionnellement
                w_opt = w_opt * adjustment_factor
                p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
                p_ret = (w_opt @ (exp_raw - pd.Series(applied_fees))) - ((np.sum(w_opt)-1) * borrow_cost if np.sum(w_opt) > 1 else 0)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net Final", f"{p_ret:.2%}")
            m2.metric("Volatilité Pilotée", f"{p_vol:.2%}", delta=f"Cible: {vol_target:.1%}")
            m3.metric("Levier Effectif", f"{np.sum(w_opt):.2f}x")
            m4.metric("Status Overlay", deleveraging_status)

            if deleveraging_status == "ACTIF":
                st.warning(f"⚠️ Levier réduit de { (1-adjustment_factor):.1%} pour respecter votre cible de volatilité.")

            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Asset Mix Post-Overlay"), use_container_width=True)
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2 if p_vol > 0 else 1)
            c2.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Risk Contribution (%)"), use_container_width=True)

    with tab_risk:
        st.header("📈 Matrice de Corrélation Historique")
        st.plotly_chart(px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=800), use_container_width=True)

except Exception as e: st.error(f"Erreur technique : {e}")
