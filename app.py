import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION INITIALE ---
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

# Frais par défaut (FNB)
DEFAULT_MER = {
    "VFV.TO": 0.0009, "VXC.TO": 0.0021, "VEE.TO": 0.0024, "ZGI.TO": 0.0061,
    "VRE.TO": 0.0039, "DBC": 0.0085, "XSU.TO": 0.0033, "VAB.TO": 0.0009,
    "XHY.TO": 0.0061, "XCB.TO": 0.0018, "PSA.TO": 0.0015
}

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    raw_data = yf.download(list(tickers.values()), period="10y", interval="1mo")
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
    return data.pct_change().dropna().rename(columns={v: k for k, v in tickers.items()})

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

# --- 2. INTERFACE SIDEBAR ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
tab_main, tab_arch = st.tabs(["📊 Optimisation et Analyse", "🔍 Architecture et Frais"])

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    
    st.header("🔮 Anticipations (CMA)")
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])
    user_rets, user_vols = {}, {}
    if mode_cma == "Manuel":
        for asset in TICKERS_DICT.keys():
            user_rets[asset] = st.number_input(f"Rend. {asset} %", value=7.0, step=0.1, key=f"r_{asset}") / 100
            user_vols[asset] = st.number_input(f"Vol. {asset} %", value=12.0, step=0.1, key=f"v_{asset}") / 100

    # --- NOUVELLE SECTION : AJUSTEMENT MANUEL DES FRAIS ---
    st.header("💸 Frais Manuels (RFG/MER)")
    st.info("Ajustez les frais réels (ex: frais de gestion des fonds privés).")
    custom_fees = {}
    for asset, ticker in TICKERS_DICT.items():
        default_val = DEFAULT_MER[ticker] * 100
        custom_fees[asset] = st.number_input(f"Frais {asset} (%)", value=default_val, step=0.05, key=f"fee_{asset}") / 100

with tab_arch:
    st.header("Comparaison des Frais de Gestion")
    fees_comparison = pd.DataFrame([
        {"Classe": k, "Ticker": TICKERS_DICT[k], "Frais FNB (%)": f"{DEFAULT_MER[TICKERS_DICT[k]]:.3%}", "Frais Appliqués (%)": f"{custom_fees[k]:.3%}"} 
        for k in TICKERS_DICT.keys()
    ])
    st.table(fees_comparison)

with tab_main:
    with st.expander("📖 Lexique et Guide des Paramètres"):
        st.markdown("* **Frais Appliqués :** Frais soustraits du rendement brut. Pour les actifs privés, on utilise souvent entre 1.0% et 2.0%.")

    st.header("📊 Politique de Placement (Min/Max)")
    asset_bounds = {}
    cols = st.columns(4)
    for i, asset in enumerate(TICKERS_DICT.keys()):
        with cols[i % 4]:
            b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"nmin_{asset}") / 100
            b_max = st.number_input(f"Max {asset} %", 0, 100, 25, key=f"nmax_{asset}") / 100
            asset_bounds[asset] = (b_min, b_max)

    try:
        hist_data = get_market_data(TICKERS_DICT)
        rfr = (1 + hist_data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)

        if mode_cma == "Manuel":
            exp_rets_raw = pd.Series(user_rets)
            v_diag = np.diag([user_vols[asset] for asset in hist_data.columns])
            adj_cov_base = pd.DataFrame(v_diag @ hist_data.corr().values @ v_diag, index=hist_data.columns, columns=hist_data.columns)
        else:
            exp_rets_raw = hist_data.mean() * 12
            adj_cov_base = hist_data.cov() * 12

        # Application des frais manuels sur le rendement attendu
        exp_rets = exp_rets_raw - pd.Series(custom_fees)
        adj_cov = desmooth_cov(adj_cov_base, alpha, ILLIQUID_ASSETS)
        w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

        if w_opt is not None:
            port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
            port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
            weighted_mer = np.sum(w_opt * pd.Series(custom_fees))
            carry = (w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net Finaux", f"{port_ret:.2%}")
            m2.metric("Volatilité", f"{port_vol:.2%}")
            m3.metric("Frais Appliqués", f"{weighted_mer:.2%}")
            m4.metric("Carry Levier", f"{carry:+.2%}")

            # Frontière Efficiente
            t_range = np.linspace(max(0.02, exp_rets.min()), min(0.12, exp_rets.max()*lev_max), 10)
            f_vols, f_rets = [], []
            for r in t_range:
                wt = optimize_portfolio(exp_rets, adj_cov, lev_max, r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)
                if wt is not None:
                    f_rets.append(wt @ exp_rets - (np.sum(wt)-1)*borrow_cost)
                    f_vols.append(np.sqrt(wt.T @ adj_cov @ wt))
            
            fig_ef = px.line(x=f_vols, y=f_rets, labels={'x':'Risque','y':'Rendement Net'}, title="Frontière Efficiente (Après Frais)")
            fig_ef.add_scatter(x=[port_vol], y=[port_ret], name="Ma Sélection", marker=dict(size=12, color='red'))
            st.plotly_chart(fig_ef, use_container_width=True)

            cola, colb = st.columns(2)
            with cola: st.plotly_chart(px.pie(values=w_opt, names=hist_data.columns, title="Allocation Capital", hole=0.4), use_container_width=True)
            with colb: st.plotly_chart(px.bar(x=hist_data.columns, y=w_opt * pd.Series(custom_fees) * 100, title="Coût par Actif (bps)"), use_container_width=True)
        else:
            st.error("⚠️ Pas de solution trouvée. Vérifiez vos contraintes.")
    except Exception as e:
        st.error(f"Erreur : {e}")
