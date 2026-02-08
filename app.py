import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Pension Canada", layout="wide")

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

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, vol_cap=None):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    
    S = (cov_matrix.values + cov_matrix.values.T) / 2 + np.eye(n) * 1e-6
    risk = cp.quad_form(w, cp.psd_wrap(S))
    
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret, w >= 0]
    
    # AJOUT DU PLAFOND DE VOLATILITÉ
    if vol_cap:
        constraints.append(risk <= (vol_cap ** 2))
        
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

# --- 2. INTERFACE ---
st.title("🏛️ Terminal CIO : Pilotage de la Volatilité")
tab_opt, tab_risk, tab_stress, tab_lex = st.tabs(["📊 Optimisation", "📈 Analyse Risque", "⚠️ Stress Test", "🔍 Lexique"])

with st.sidebar:
    st.header("⚙️ Gouvernance")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    
    # NOUVEAU : PLAFOND DE VOLATILITÉ
    st.subheader("🛡️ Gestion du Risque")
    vol_limit = st.slider("Plafond Volatilité (%)", 5.0, 15.0, 8.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])

try:
    data = get_market_data(TICKERS_DICT)
    applied_fees = {a: st.sidebar.number_input(f"Frais {a} %", 0.2, key=f"f_{a}")/100 for a in TICKERS_DICT.keys()}

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
        exp_raw = data.mean()*12 # Simplifié pour l'exemple
        cov_base = data.cov()*12
        
        # Application Alpha
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        # Optimisation avec Plafond Vol
        w_opt = optimize_portfolio(exp_raw - pd.Series(applied_fees), cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i, vol_cap=vol_limit)
        
        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            p_ret = (w_opt @ (exp_raw - pd.Series(applied_fees))) - ((np.sum(w_opt)-1) * borrow_cost)
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Net Return", f"{p_ret:.2%}")
            m2.metric("Portfolio Vol", f"{p_vol:.2%}")
            m3.metric("Leverage Used", f"{np.sum(w_opt):.2f}x")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Asset Mix"), use_container_width=True)
            with c2:
                rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2)
                st.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Risk Contribution (%)"), use_container_width=True)
        else:
            st.warning("⚠️ Cible inatteignable avec ce plafond de volatilité. Augmentez le plafond ou baissez la cible de rendement.")

    with tab_stress:
        st.header("⚠️ Simulation de Crise (Choc de Corrélation)")
        st.write("Que se passe-t-il si toutes les corrélations montent à 0.80 lors d'un krach ?")
        stress_corr = pd.DataFrame(0.80, index=data.columns, columns=data.columns)
        np.fill_diagonal(stress_corr.values, 1.0)
        v_diag = np.diag(np.sqrt(np.diag(cov_base)))
        stress_cov = v_diag @ stress_corr @ v_diag
        
        if 'w_opt' in locals() and w_opt is not None:
            stress_vol = np.sqrt(w_opt.T @ stress_cov @ w_opt)
            st.metric("Volatilité en période de crise", f"{stress_vol:.2%}", delta=f"{stress_vol - p_vol:.2%}", delta_color="inverse")
            st.error(f"En cas de crise systémique, votre volatilité bondirait de {(stress_vol - p_vol):.2%}.")

except Exception as e: st.error(f"Erreur : {e}")
