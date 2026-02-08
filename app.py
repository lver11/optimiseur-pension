import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION ET RÉFÉRENTIEL ---
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
    risk = cp.quad_form(w, cov_matrix.values)
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret, w >= 0]
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.OSQP)
    return w.value if w.value is not None else None

# --- 2. INTERFACE ET NAVIGATION ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
tab_opt, tab_risk, tab_data = st.tabs(["📊 Optimisation", "⚖️ Risque & Matrice", "🔍 Lexique & Architecture"])

with st.sidebar:
    st.header("⚙️ Configuration Globale")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])

with tab_data:
    st.header("📖 Lexique et Guide de l'Outil")
    st.markdown("""
    * **Levier Brut :** Capacité d'emprunt (ex: 1.25x = 25% de levier sur capital propre).
    * **Alpha (Délissage) :** Correction de la volatilité des actifs non cotés (Immobilier, Dette Privée).
    * **Carry du Levier :** Rendement moyen des actifs moins le coût de l'emprunt.
    * **Total Portfolio Approach (TPA) :** Stratégie où chaque actif est choisi pour sa contribution au risque total plutôt que son rendement isolé.
    """)
    st.divider()
    st.header("🔍 Architecture des Frais (RFG/MER)")
    applied_fees = {}
    cols_f = st.columns(3)
    for i, (asset, ticker) in enumerate(TICKERS_DICT.items()):
        with cols_f[i % 3]:
            applied_fees[asset] = st.number_input(f"Frais {asset} (%)", value=DEFAULT_MER[ticker]*100, step=0.05, key=f"f_tab_{asset}") / 100

with tab_opt:
    st.header("📊 Bornes de Placement")
    asset_bounds = {}
    cols = st.columns(4)
    for i, asset in enumerate(TICKERS_DICT.keys()):
        with cols[i % 4]:
            b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"min_p_{asset}") / 100
            def_max = 0 if "Cash" in asset else (80 if "Mondiales" in asset else 40)
            b_max = st.number_input(f"Max {asset} %", 0, 100, def_max, key=f"max_p_{asset}") / 100
            asset_bounds[asset] = (b_min, b_max)

    try:
        data = get_market_data(TICKERS_DICT)
        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)

        if mode_cma == "Manuel":
            exp_raw = pd.Series([st.sidebar.number_input(f"Rend. {a} %", 0.0, 20.0, 7.0, key=f"r_s_{a}")/100 for a in TICKERS_DICT.keys()], index=TICKERS_DICT.keys())
            vols = [st.sidebar.number_input(f"Vol. {a} %", 0.0, 40.0, 12.0, key=f"v_s_{a}")/100 for a in TICKERS_DICT.keys()]
            cov_base = pd.DataFrame(np.diag(vols) @ data.corr().values @ np.diag(vols), index=data.columns, columns=data.columns)
        else:
            exp_raw = data.mean() * 12
            cov_base = data.cov() * 12

        exp_rets = exp_raw - pd.Series(applied_fees)
        for a in ILLIQUID_ASSETS:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        w_opt = optimize_portfolio(exp_rets, cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

        if w_opt is not None:
            w_opt = np.array([x if x > 0.001 else 0 for x in w_opt])
            p_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost)
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net", f"{p_ret:.2%}")
            m2.metric("Volatilité", f"{p_vol:.2%}")
            m3.metric("Carry Levier", f"{(w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost:+.2%}")
            m4.metric("Frais Totaux", f"{np.sum(w_opt * pd.Series(applied_fees)):.2%}")

            c_pie, c_ef = st.columns(2)
            with c_pie:
                df_p = pd.DataFrame({"Actif": list(TICKERS_DICT.keys()), "Poids": w_opt})
                fig = px.pie(df_p[df_p["Poids"]>0], values="Poids", names="Actif", hole=0.4, title="Allocation Réelle")
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
                fig_ef = px.line(x=f_vols, y=f_rets, title="Frontière Efficiente Net", labels={'x':'Vol','y':'Rend'})
                fig_ef.add_scatter(x=[p_vol], y=[p_ret], name="Ma Sélection", marker=dict(size=12, color='red'))
                st.plotly_chart(fig_ef, use_container_width=True)

            st.download_button("📥 Export CSV", df_p.to_csv(index=False).encode('utf-8'), "allocation.csv")
        else:
            st.error("⚠️ Pas de solution. Ajustez les bornes Max.")
    except Exception as e: st.error(f"Erreur : {e}")

with tab_risk:
    if 'data' in locals():
        st.header("📈 Matrice de Corrélation et Contribution au Risque")
        st.plotly_chart(px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
        if 'w_opt' in locals() and w_opt is not None:
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2)
            st.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Contribution au Risque (%)"), use_container_width=True)
