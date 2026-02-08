import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION ET DONNÉES ---
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

@st.cache_data
def get_market_data(tickers):
    raw_data = yf.download(list(tickers.values()), period="10y", interval="1mo")
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
    returns = data.pct_change().dropna()
    return returns.rename(columns={v: k for k, v in tickers.items()})

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

# --- 2. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
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
    st.header("Frais de Gestion (RFG/MER) et Proxys")
    st.table(pd.DataFrame([{"Classe": k, "Ticker": v, "Description": PROXY_DATA[v]['desc'], "RFG": f"{PROXY_DATA[v]['mer']:.2%}"} for k, v in TICKERS_DICT.items()]))

with tab_main:
    with st.expander("📖 Lexique et Guide des Paramètres"):
        st.markdown("""
        * **Levier Brut Max :** Taille maximale du portefeuille incluant l'emprunt.
        * **Max Alternatifs :** Plafond cumulé pour les actifs illiquides (Immo, Infra, Dette).
        * **Carry Net du Levier :** Différence entre le rendement des actifs et le coût de l'intérêt.
        * **Alpha :** Correction de la volatilité pour les actifs non cotés.
        * **Frais Totaux (RFG) :** Moyenne pondérée des frais des instruments utilisés.
        """)

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
            exp_rets = pd.Series(user_rets)
            v_diag = np.diag([user_vols[asset] for asset in hist_data.columns])
            adj_cov_base = pd.DataFrame(v_diag @ hist_data.corr().values @ v_diag, index=hist_data.columns, columns=hist_data.columns)
        else:
            exp_rets = hist_data.mean() * 12
            adj_cov_base = hist_data.cov() * 12

        adj_cov = desmooth_cov(adj_cov_base, alpha, ILLIQUID_ASSETS)
        w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

        if w_opt is not None:
            # Calculs finaux
            port_ret = w_opt @ exp_rets - (np.sum(w_opt)-1)*borrow_cost
            port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
            mer_pondere = np.sum(w_opt * [PROXY_DATA[TICKERS_DICT[a]]['mer'] for a in hist_data.columns])
            carry = (w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net", f"{port_ret:.2%}")
            m2.metric("Volatilité", f"{port_vol:.2%}")
            m3.metric("Frais (RFG)", f"{mer_pondere:.2%}")
            m4.metric("Carry Levier", f"{carry:+.2%}")

            # --- FRONTIÈRE EFFICIENTE ---
            st.subheader("📈 Frontière Efficiente")
            t_range = np.linspace(max(0.02, exp_rets.min()), min(0.12, exp_rets.max()*lev_max), 10)
            f_vols, f_rets = [], []
            for r in t_range:
                wt = optimize_portfolio(exp_rets, adj_cov, lev_max, r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)
                if wt is not None:
                    f_rets.append(wt @ exp_rets - (np.sum(wt)-1)*borrow_cost)
                    f_vols.append(np.sqrt(wt.T @ adj_cov @ wt))
            fig_ef = px.line(x=f_vols, y=f_rets, labels={'x':'Risque','y':'Rendement'}, title="Frontière Risque-Rendement")
            fig_ef.add_scatter(x=[port_vol], y=[port_ret], name="Ma Sélection", marker=dict(size=12, color='red'))
            st.plotly_chart(fig_ef, use_container_width=True)

            cola, colb = st.columns(2)
            with cola: st.plotly_chart(px.pie(values=w_opt, names=hist_data.columns, title="Allocation Capital", hole=0.4), use_container_width=True)
            with colb: st.plotly_chart(px.bar(x=hist_data.columns, y=w_opt * [PROXY_DATA[TICKERS_DICT[a]]['mer'] for a in hist_data.columns] * 100, title="Impact des Frais (bps)"), use_container_width=True)
        else:
            st.error("⚠️ Pas de solution.")
    except Exception as e:
        st.error(f"Erreur : {e}")
