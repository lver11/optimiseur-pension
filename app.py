import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS ET PROXYS ---
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

# Données des frais (MER)
PROXY_DATA = {
    "VFV.TO": {"desc": "Vanguard S&P 500", "mer": 0.0009},
    "VXC.TO": {"desc": "Vanguard Global All Cap", "mer": 0.0021},
    "VEE.TO": {"desc": "Vanguard Emerging Markets", "mer": 0.0024},
    "ZGI.TO": {"desc": "BMO Global Infrastructure", "mer": 0.0061},
    "VRE.TO": {"desc": "Vanguard Cdn REIT", "mer": 0.0039},
    "DBC": {"desc": "Invesco DB Commodity", "mer": 0.0085},
    "XSU.TO": {"desc": "iShares Russell 2000", "mer": 0.0033},
    "VAB.TO": {"desc": "Vanguard Cdn Aggregate", "mer": 0.0009},
    "XHY.TO": {"desc": "iShares US High Yield", "mer": 0.0061},
    "XCB.TO": {"desc": "iShares Cdn Corporate", "mer": 0.0018},
    "PSA.TO": {"desc": "Purpose High Interest", "mer": 0.0015}
}

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

# --- 2. FONCTIONS DE CALCUL ---
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

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret]
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- 3. INTERFACE UTILISATEUR ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")

# Onglets
tab_main, tab_arch = st.tabs(["📊 Optimisation et Analyse", "🔍 Architecture et Frais"])

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])
    user_rets, user_vols = {}, {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            user_rets[asset] = st.number_input(f"Rend. {asset} %", value=7.0, step=0.1) / 100
            user_vols[asset] = st.number_input(f"Vol. {asset} %", value=12.0, step=0.1) / 100

with tab_arch:
    st.header("Architecture des Données et Frais (RFG/MER)")
    proxy_df = pd.DataFrame([{"Classe": k, "Ticker": v, "Description": PROXY_DATA[v]['desc'], "RFG (MER)": f"{PROXY_DATA[v]['mer']:.2%}"} for k, v in TICKERS_DICT.items()])
    st.table(proxy_df)

with tab_main:
    # --- RÉINTÉGRATION DU LEXIQUE ---
    with st.expander("📖 Lexique et Guide des Paramètres"):
        st.markdown("""
        ### Paramètres Clés
        * **Levier Brut :** Capacité d'emprunt pour augmenter l'exposition (ex: 1.25x = 25% de levier).
        * **Alpha (Délissage) :** Corrige la sous-estimation du risque des actifs privés qui ne sont pas évalués quotidiennement.
        * **Spread (bps) :** Coût d'emprunt additionnel (100 bps = 1.00%).
        ### Indicateurs
        * **RFG Pondéré :** Moyenne des frais de gestion de tous les FNB selon votre allocation.
        * **Ratio de Sharpe :** Performance par unité de risque (plus il est haut, mieux c'est).
        * **Unhedged :** Exposition totale à la devise étrangère (protection naturelle en cas de crise).
        """)

    st.header("📊 Politique de Placement (Bornes Min/Max)")
    asset_bounds = {}
    cols = st.columns(4)
    for i, asset in enumerate(TICKERS_DICT.keys()):
        with cols[i % 4]:
            b_min = st.number_input(f"Min {asset} (%)", 0, 100, 0, key=f"min_{asset}") / 100
            b_max = st.number_input(f"Max {asset} (%)", 0, 100, 25, key=f"max_{asset}") / 100
            asset_bounds[asset] = (b_min, b_max)

    try:
        hist_data = get_market_data(TICKERS_DICT)
        rfr = (1 + hist_data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)

        if mode_cma == "Manuel":
            exp_rets = pd.Series(user_rets)
            vols_diag = np.diag([user_vols[asset] for asset in hist_data.columns])
            manual_cov = vols_diag @ hist_data.corr().values @ vols_diag
            adj_cov_base = pd.DataFrame(manual_cov, index=hist_data.columns, columns=hist_data.columns)
        else:
            exp_rets = hist_data.mean() * 12
            adj_cov_base = hist_data.cov() * 12

        adj_cov = desmooth_cov(adj_cov_base, alpha, ILLIQUID_ASSETS)
        w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

        if w_opt is not None:
            port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
            port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
            mer_list = [PROXY_DATA[TICKERS_DICT[asset]]['mer'] for asset in hist_data.columns]
            weighted_mer = np.sum(w_opt * mer_list)
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net", f"{port_ret:.2%}")
            m2.metric("Volatilité Ajustée", f"{port_vol:.2%}")
            m3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")
            m4.metric("Frais Totaux (RFG)", f"{weighted_mer:.2%}")

            cola, colb = st.columns(2)
            with cola:
                st.plotly_chart(px.pie(values=w_opt, names=hist_data.columns, title="Allocation Capital", hole=0.4), use_container_width=True)
            with colb:
                fee_impact = w_opt * mer_list
                st.plotly_chart(px.bar(x=hist_data.columns, y=fee_impact*100, title="Coût par Actif (bps)"), use_container_width=True)

            st.divider()
            st.subheader("🏁 Performance Historique vs 60/40")
            bench = (hist_data["Actions Mondiales (Unhedged)"] * 0.60) + (hist_data["Obligations Can"] * 0.40)
            comp_df = pd.DataFrame({"Optimisé": (1 + hist_data.dot(w_opt)).cumprod() * 100, "60/40": (1 + bench).cumprod() * 100}, index=hist_data.index)
            st.line_chart(comp_df)
        else:
            st.error("⚠️ Aucune solution trouvée. Ajustez vos bornes.")
    except Exception as e:
        st.error(f"Erreur technique : {e}")
