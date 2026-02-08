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

# Détails techniques des instruments (Source: Sites des émetteurs - Approx.)
PROXY_DETAILS = [
    {"Classe": "Actions US (Unhedged)", "Ticker": "VFV.TO", "Description": "Vanguard S&P 500 ETF", "RFG (MER)": "0.09%"},
    {"Classe": "Actions Mondiales (Unhedged)", "Ticker": "VXC.TO", "Description": "Vanguard Global All Cap ETF", "RFG (MER)": "0.21%"},
    {"Classe": "Marchés Émergents", "Ticker": "VEE.TO", "Description": "Vanguard Emerging Markets ETF", "RFG (MER)": "0.24%"},
    {"Classe": "Infrastructures", "Ticker": "ZGI.TO", "Description": "BMO Global Infrastructure Index ETF", "RFG (MER)": "0.61%"},
    {"Classe": "Immobilier Listé", "Ticker": "VRE.TO", "Description": "Vanguard Cdn Capped REIT ETF", "RFG (MER)": "0.39%"},
    {"Classe": "Matières Premières", "Ticker": "DBC", "Description": "Invesco DB Commodity Tracking", "RFG (MER)": "0.85%"},
    {"Classe": "Petites Caps US (Unhedged)", "Ticker": "XSU.TO", "Description": "iShares Russell 2000 (CAD Unhedged)", "RFG (MER)": "0.33%"},
    {"Classe": "Obligations Can", "Ticker": "VAB.TO", "Description": "Vanguard Cdn Aggregate Bond ETF", "RFG (MER)": "0.09%"},
    {"Classe": "Dette Privée (Proxy)", "Ticker": "XHY.TO", "Description": "iShares US High Yield (CAD Hedged)", "RFG (MER)": "0.61%"},
    {"Classe": "Hypothèques Comm. (Proxy)", "Ticker": "XCB.TO", "Description": "iShares Cdn Corporate Bond ETF", "RFG (MER)": "0.18%"},
    {"Classe": "Cash (RFR)", "Ticker": "PSA.TO", "Description": "Purpose High Interest Savings ETF", "RFG (MER)": "0.15%"}
]

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
    asset_names = list(returns_series.index)
    constraints = [cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret]
    for i, name in enumerate(asset_names):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    ill_idx = [i for i, name in enumerate(asset_names) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value if w.value is not None else None

# --- 3. INTERFACE UTILISATEUR ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
tab_main, tab_arch = st.tabs(["📊 Optimisation et Stratégie", "🔍 Architecture des Données (Proxys)"])

with st.sidebar:
    st.header("⚙️ Paramètres")
    lev_max = st.slider("Levier Brut Maximum", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible de Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    st.header("⚖️ Limites Alternatifs")
    max_i = st.slider("Plafond Alternatifs Globaux (%)", 10, 80, 45) / 100
    st.header("💳 Financement")
    spread_bps = st.number_input("Spread sur levier (bps)", value=120, step=10)
    st.header("🔮 Anticipations (CMA)")
    mode_cma = st.radio("Source :", ["Historique", "Manuel"])
    user_rets, user_vols = {}, {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            user_rets[asset] = st.number_input(f"Rend. {asset} %", value=7.0, step=0.1) / 100
            user_vols[asset] = st.number_input(f"Vol. {asset} %", value=12.0, step=0.1) / 100

with tab_arch:
    st.header("Architecture des Données et Instruments de Référence")
    st.markdown("Détails des FNB (ETF) utilisés comme bases de calcul pour les corrélations et rendements.")
    st.table(pd.DataFrame(PROXY_DETAILS))
    st.info("💡 Note : Les rendements affichés dans l'application intègrent déjà les frais de gestion (RFG) des FNB.")

with tab_main:
    with st.expander("📖 Glossaire des paramètres"):
        st.markdown("* **Levier :** Taille totale du portefeuille divisée par le capital propre. \n* **Alpha :** Facteur de correction pour 'révéler' la volatilité cachée des actifs illiquides. \n* **Spread :** Marge ajoutée par la banque au taux de base pour votre ligne de crédit.")
    
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
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement Net", f"{port_ret:.2%}")
            c2.metric("Volatilité Ajustée", f"{port_vol:.2%}")
            c3.metric("Ratio de Sharpe", f"{(port_ret-rfr)/port_vol:.2f}")

            # Graphiques
            cola, colb = st.columns(2)
            with cola:
                st.plotly_chart(px.pie(values=w_opt, names=hist_data.columns, title="Allocation du Capital", hole=0.4), use_container_width=True)
            with colb:
                risk_contrib = (w_opt * (adj_cov @ w_opt)) / (port_vol**2)
                st.plotly_chart(px.pie(values=risk_contrib, names=hist_data.columns, title="Allocation du Risque", hole=0.4), use_container_width=True)

            # Performance historique
            st.divider()
            st.subheader("🏁 Performance Historique vs 60/40")
            bench = (hist_data["Actions Mondiales (Unhedged)"] * 0.60) + (hist_data["Obligations Can"] * 0.40)
            comp_df = pd.DataFrame({"Optimisé": (1 + hist_data.dot(w_opt)).cumprod() * 100, "60/40": (1 + bench).cumprod() * 100}, index=hist_data.index)
            st.line_chart(comp_df)
        else:
            st.error("⚠️ Impossible de trouver une solution respectant ces bornes.")
    except Exception as e:
        st.error(f"Erreur technique : {e}")
