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

# --- 3. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")

tab_main, tab_arch = st.tabs(["📊 Optimisation et Analyse", "🔍 Architecture et Frais"])

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs Globaux (%)", 10, 80, 45) / 100
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
    with st.expander("📖 Lexique et Guide des Paramètres"):
        st.markdown("""
        ### Analyse du Levier
        * **Rendement Brut du Levier :** Rendement généré par les actifs achetés avec l'argent emprunté.
        * **Coût du Financement :** Taux d'intérêt total payé sur l'emprunt (Taux sans risque + Spread).
        * **Carry Net (Levier) :** La différence entre le rendement des actifs et le coût de l'emprunt. Si c'est positif, le levier crée de la valeur.
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
            # --- CALCULS DE PERFORMANCE ---
            lev_size = np.sum(w_opt) - 1
            gross_port_ret = w_opt @ exp_rets
            total_interest_cost = lev_size * borrow_cost
            port_ret_net = gross_port_ret - total_interest_cost
            
            # Carry du levier (Rendement pondéré des actifs / somme des poids - coût emprunt)
            avg_asset_yield = (w_opt @ exp_rets) / np.sum(w_opt)
            leverage_carry = avg_asset_yield - borrow_cost
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net", f"{port_ret_net:.2%}")
            m2.metric("Volatilité Ajustée", f"{np.sqrt(w_opt.T @ adj_cov @ w_opt):.2%}")
            m3.metric("Ratio de Sharpe", f"{(port_ret_net-rfr)/np.sqrt(w_opt.T @ adj_cov @ w_opt):.2f}")
            m4.metric("Carry Net du Levier", f"{leverage_carry:+.2%}", help="Différence entre le rendement moyen des actifs et le coût de l'emprunt.")

            # --- SECTION ANALYSE DU LEVIER ---
            st.subheader("🕵️ Analyse de la Structure de Rendement")
            c_lev1, c_lev2 = st.columns(2)
            with c_lev1:
                st.write("**Décomposition du Rendement ($)**")
                lev_data = pd.DataFrame({
                    "Composante": ["Rendement des Actifs", "Coût de l'Intérêt"],
                    "Valeur (%)": [gross_port_ret * 100, -total_interest_cost * 100]
                })
                st.plotly_chart(px.bar(lev_data, x="Composante", y="Valeur (%)", color="Composante", 
                                       color_discrete_map={"Rendement des Actifs": "green", "Coût de l'Intérêt": "red"}), use_container_width=True)
            with c_lev2:
                st.write("**Efficacité du Levier**")
                st.info(f"""
                * **Taille de l'emprunt :** {lev_size:.1%} du capital propre.
                * **Coût total de l'emprunt :** {borrow_cost:.2%} (Taux {rfr:.2%} + Spread {spread_bps}bps).
                * **Seuil de rentabilité (Hurdle) :** Vos actifs doivent générer plus de **{borrow_cost:.2%}** pour que le levier soit profitable.
                """)

            cola, colb = st.columns(2)
            with cola:
                st.plotly_chart(px.pie(values=w_opt, names=hist_data.columns, title="Allocation Capital", hole=0.4), use_container_width=True)
            with colb:
                risk_contrib = (w_opt * (adj_cov @ w_opt)) / (w_opt.T @ adj_cov @ w_opt)
                st.plotly_chart(px.pie(values=risk_contrib, names=hist_data.columns, title="Allocation Risque", hole=0.4), use_container_width=True)

            st.divider()
            st.subheader("🏁 Performance Historique vs 60/40")
            bench = (hist_data["Actions Mondiales (Unhedged)"] * 0.60) + (hist_data["Obligations Can"] * 0.40)
            comp_df = pd.DataFrame({"Optimisé": (1 + hist_data.dot(w_opt)).cumprod() * 100, "60/40": (1 + bench).cumprod() * 100}, index=hist_data.index)
            st.line_chart(comp_df)
        else:
            st.error("⚠️ Aucune solution trouvée. Ajustez vos bornes.")
    except Exception as e:
        st.error(f"Erreur technique : {e}")
