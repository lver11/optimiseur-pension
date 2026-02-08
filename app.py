import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Pension Canada", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS ---
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

DEFAULT_MER = {
    "VFV.TO": 0.0009, "VXC.TO": 0.0021, "VEE.TO": 0.0024, "VWOB": 0.0020,
    "ZGI.TO": 0.0061, "VRE.TO": 0.0039, "DBC": 0.0085, "XSU.TO": 0.0033, 
    "VAB.TO": 0.0009, "XHY.TO": 0.0061, "XCB.TO": 0.0018, "PSA.TO": 0.0015
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
    st.header("⚙️ Gouvernance")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    
    st.header("🔮 Capital Market Assumptions")
    mode_cma = st.radio("Source des données :", ["Historique", "Manuel"])
    user_rets, user_vols = {}, {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            c1, c2 = st.columns(2)
            user_rets[asset] = c1.number_input(f"Rend. {asset} %", 7.0, key=f"r_{asset}")/100
            user_vols[asset] = c2.number_input(f"Vol. {asset} %", 12.0, key=f"v_{asset}")/100

# --- 3. ONGLETS PRINCIPAUX ---
tab_opt, tab_risk, tab_hist, tab_lex = st.tabs(["📊 Optimisation", "📈 Corrélations", "🕒 Backtest", "🔍 Lexique & Frais"])

try:
    data = get_market_data(TICKERS_DICT)
    
    with tab_lex:
        st.header("📖 Glossaire Institutionnel")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Gestion du Risque et Actifs Privés
            * **Alpha (Délissage) :** Ajustement statistique pour les actifs illiquides (Immo, Dette Privée). Les prix étant basés sur des évaluations périodiques, la volatilité historique est sous-estimée. Un Alpha plus bas augmente la volatilité et les corrélations simulées.
            * **TPA (Total Portfolio Approach) :** Méthode de gestion unifiée où l'on optimise le risque total du fonds plutôt que d'allouer par classes d'actifs isolées (silos).
            * **Max Drawdown (MDD) :** La perte maximale enregistrée entre un sommet et un creux historique. Crucial pour tester la survie du levier.
            """)
        with col2:
            st.markdown("""
            ### Ingénierie du Levier
            * **Levier Brut Max :** Taille de l'exposition totale divisée par le capital propre.
            * **Carry du Levier :** Revenu net généré par l'emprunt (Rendement de l'actif - Coût du financement).
            * **Spread Levier :** Prime payée au-dessus du taux sans risque (PSA.TO) pour financer le levier.
            """)
        st.divider()
        st.subheader("Architecture des Frais (MER/RFG)")
        applied_fees = {a: st.number_input(f"Frais {a} %", DEFAULT_MER[TICKERS_DICT[a]]*100, step=0.01, key=f"f_{a}")/100 for a in TICKERS_DICT.keys()}

    with tab_risk:
        st.header("📈 Matrice de Corrélation (Large Format)")
        fig_corr = px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=800)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_opt:
        st.header("📊 Paramétrage des Bornes Tactiques")
        asset_bounds = {}
        cols = st.columns(4)
        for i, asset in enumerate(TICKERS_DICT.keys()):
            with cols[i % 4]:
                b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"min_{asset}")/100
                b_max = st.number_input(f"Max {asset} %", 0, 100, 40 if "Cash" not in asset else 0, key=f"max_{asset}")/100
                asset_bounds[asset] = (b_min, b_max)

        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        exp_raw = pd.Series(user_rets) if mode_cma == "Manuel" else data.mean()*12
        cov_base = data.cov()*12
        if mode_cma == "Manuel":
            v_diag = np.diag([user_vols[a] for a in TICKERS_DICT.keys()])
            cov_base = pd.DataFrame(v_diag @ data.corr().values @ v_diag, index=data.columns, columns=data.columns)
        
        exp_rets = exp_raw - pd.Series(applied_fees)
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        w_opt = optimize_portfolio(exp_rets, cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i)
        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            p_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost)
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Net Return", f"{p_ret:.2%}")
            m2.metric("Portfolio Vol", f"{p_vol:.2%}")
            m3.metric("Leverage Carry", f"{(w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost:+.2%}")
            m4.metric("Total Fees", f"{np.sum(w_opt * pd.Series(applied_fees)):.2%}")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Asset Mix"), use_container_width=True)
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2)
            c2.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Risk Contribution (%)"), use_container_width=True)

    with tab_hist:
        if 'w_opt' in locals() and w_opt is not None:
            st.header("🕒 Performance Historique Simulée")
            lev_s = np.sum(w_opt) - 1
            port_m = data.dot(w_opt) - (lev_s * (borrow_cost/12)) - np.sum(w_opt * (pd.Series(applied_fees)/12))
            cum_port = (1 + port_m).cumprod() * 100000
            st.line_chart(cum_port)
            st.subheader("Analyse du Drawdown")
            drawdown = (cum_port - cum_port.cummax()) / cum_port.cummax()
            st.area_chart(drawdown)
            st.metric("Pire baisse (10 ans)", f"{drawdown.min():.2%}")

except Exception as e: st.error(f"Erreur technique : {e}")
