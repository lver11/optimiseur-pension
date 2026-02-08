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

@st.cache_data
def get_market_data(tickers):
    keys = list(tickers.keys())
    raw = yf.download([tickers[k] for k in keys], period="10y", interval="1mo")
    df = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df[keys].pct_change().dropna()

# --- 2. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])

st.header("📊 Bornes de l'Allocation")
asset_bounds = {}
cols = st.columns(4)
for i, asset in enumerate(TICKERS_DICT.keys()):
    with cols[i % 4]:
        st.write(f"**{asset}**")
        b_min = st.number_input("Min %", 0, 100, 0, key=f"min_{asset}") / 100
        # Défaut : 0 pour Cash, 80 pour Actions Mondiales, 40 pour le reste
        default_val = 0 if "Cash" in asset else (80 if "Mondiales" in asset else 40)
        b_max = st.number_input("Max %", 0, 100, default_val, key=f"max_{asset}") / 100
        asset_bounds[asset] = (b_min, b_max)

try:
    data = get_market_data(TICKERS_DICT)
    rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + (spread_bps / 10000)

    # Paramètres financiers (Frais, Rendements, Vols)
    # Nous utilisons des sliders/inputs avec clés uniques pour éviter les bugs
    fees = pd.Series([st.sidebar.number_input(f"Frais {a} %", 0.0, 5.0, 0.2, key=f"f_{a}")/100 for a in TICKERS_DICT.keys()], index=TICKERS_DICT.keys())

    if mode_cma == "Manuel":
        exp_raw = pd.Series([st.sidebar.number_input(f"Rend. {a} %", 0.0, 20.0, 7.0, key=f"r_{a}")/100 for a in TICKERS_DICT.keys()], index=TICKERS_DICT.keys())
        v_diag = np.diag([st.sidebar.number_input(f"Vol. {a} %", 0.0, 40.0, 12.0, key=f"v_{a}")/100 for a in TICKERS_DICT.keys()])
        cov_base = pd.DataFrame(v_diag @ data.corr().values @ v_diag, index=data.columns, columns=data.columns)
    else:
        exp_raw = data.mean() * 12
        cov_base = data.cov() * 12

    exp_rets = exp_raw - fees
    
    # Délissage des alternatifs
    ill_list = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]
    for asset in ill_list:
        if asset in cov_base.index:
            cov_base.loc[asset, :] *= (1/alpha)
            cov_base.loc[:, asset] *= (1/alpha)

    # Optimisation (Solveur OSQP)
    n = len(TICKERS_DICT)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ exp_rets.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_base.values)
    
    constraints = [cp.sum(w) <= lev_max, cp.sum(w) >= 1.0, net_return >= target_r, w >= 0]
    for i, name in enumerate(TICKERS_DICT.keys()):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
    constraints.append(cp.sum(w[[i for i, a in enumerate(TICKERS_DICT.keys()) if a in ill_list]]) <= max_i)

    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.OSQP)

    if w.value is not None:
        w_final = np.array([x if x > 0.001 else 0 for x in w.value])
        
        st.divider()
        st.subheader("🎯 Résultat de l'Allocation Optimale")
        
        # Graphique avec NOMS et POURCENTAGES affichés
        df_plot = pd.DataFrame({"Actif": list(TICKERS_DICT.keys()), "Poids": w_final})
        df_plot = df_plot[df_plot["Poids"] > 0]
        
        fig = px.pie(df_plot, values="Poids", names="Actif", 
                     hole=0.4, template="plotly_white")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Rendement Net Estimé", f"{(w_final @ exp_rets - (np.sum(w_final)-1)*borrow_cost):.2%}")
        m2.metric("Volatilité (Risque)", f"{np.sqrt(w_final.T @ cov_base @ w_final):.2%}")
        m3.metric("Levier Brut", f"{np.sum(w_final):.2f}x")
    else:
        st.error("⚠️ Pas de solution. Essayez d'augmenter les plafonds Max (ex: Obligations ou Actions Mondiales).")

except Exception as e:
    st.error(f"Erreur : {e}")
