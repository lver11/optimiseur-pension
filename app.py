import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

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

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    S = (cov_matrix.values + cov_matrix.values.T) / 2 + np.eye(n) * 1e-5
    risk = cp.quad_form(w, cp.psd_wrap(S))
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret, w >= 0]
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve(solver=cp.OSQP)
    return w.value if w.value is not None else None

# --- 2. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
tab_opt, tab_risk, tab_lex = st.tabs(["📊 Optimisation", "📈 Corrélations & Risque", "🔍 Lexique"])

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])

with tab_lex:
    st.header("📖 Lexique")
    st.markdown("* **TPA :** Total Portfolio Approach.")
    applied_fees = {a: st.number_input(f"Frais {a} %", DEFAULT_MER[TICKERS_DICT[a]]*100, key=f"f_{a}")/100 for a in TICKERS_DICT.keys()}

with tab_risk:
    try:
        data = get_market_data(TICKERS_DICT)
        st.header("📈 Matrice de Corrélation Historique (Détaillée)")
        
        # --- AGRANDISSEMENT DE LA MATRICE ---
        corr_matrix = data.corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r',
            labels=dict(color="Corrélation"),
            aspect="auto"
        )
        # Forcer la hauteur à 800px et ajuster les marges pour la lisibilité
        fig_corr.update_layout(height=800, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("💡 Une matrice large permet de mieux identifier les îlots de diversification, particulièrement entre les actifs 'Unhedged' et les obligations.")
    except Exception as e:
        st.error(f"Erreur data : {e}")

with tab_opt:
    st.header("📊 Bornes de Placement")
    asset_bounds = {}
    cols = st.columns(4)
    for i, asset in enumerate(TICKERS_DICT.keys()):
        with cols[i % 4]:
            b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"min_{asset}") / 100
            def_max = 0 if "Cash" in asset else 40
            b_max = st.number_input(f"Max {asset} %", 0, 100, def_max, key=f"max_{asset}") / 100
            asset_bounds[asset] = (b_min, b_max)

    try:
        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        exp_rets = (data.mean() * 12) - pd.Series(applied_fees)
        cov_base = data.cov() * 12
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        w_opt = optimize_portfolio(exp_rets, cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)
        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            st.divider()
            p_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost)
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement Net", f"{p_ret:.2%}")
            c2.metric("Volatilité", f"{p_vol:.2%}")
            c3.metric("Carry Levier", f"{(w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost:+.2%}")

            c_pie, c_ef = st.columns(2)
            with c_pie:
                df_p = pd.DataFrame({"Actif": list(TICKERS_DICT.keys()), "Poids": w_opt})
                fig = px.pie(df_p[df_p["Poids"]>0], values="Poids", names="Actif", hole=0.4, title="Allocation")
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            with c_ef:
                t_range = np.linspace(0.04, 0.12, 10)
                f_vols, f_rets = [], []
                for r in t_range:
                    wt = optimize_portfolio(exp_rets, cov_base, lev_max, r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)
                    if wt is not None:
                        f_rets.append((wt @ exp_rets) - ((np.sum(wt)-1)*borrow_cost))
                        f_vols.append(np.sqrt(wt.T @ cov_base @ wt))
                st.plotly_chart(px.line(x=f_vols, y=f_rets, title="Frontière Efficiente", labels={'x':'Vol','y':'Rend'}), use_container_width=True)
    except Exception as e: st.error(f"Erreur : {e}")
