import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION STRICTE ---
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
    t_list = [tickers[k] for k in keys]
    raw = yf.download(t_list, period="10y", interval="1mo")
    df = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df[keys].pct_change().dropna()

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, illiquid_list):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    risk = cp.quad_form(w, cov_matrix.values)
    
    # CONTRAINTES
    constraints = [
        cp.sum(w) <= lev_limit, 
        cp.sum(w) >= 1.0, 
        net_return >= target_ret,
        w >= 0 # Long only
    ]
    
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
        
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.OSQP)
    return w.value if w.value is not None else None

# --- 2. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    
    st.header("🔮 Rendements (CMA)")
    mode_cma = st.radio("Source :", ["Historique", "Manuel"])
    
    manual_rets = {}
    manual_vols = {}
    for asset in TICKERS_DICT.keys():
        if mode_cma == "Manuel":
            manual_rets[asset] = st.number_input(f"Rend. {asset} %", 7.0, key=f"r_in_{asset}") / 100
            manual_vols[asset] = st.number_input(f"Vol. {asset} %", 12.0, key=f"v_in_{asset}") / 100
        
        # Saisie des frais déplacée ici pour être globale
        st.sidebar.markdown(f"**Frais {asset}**")
        fee_val = st.sidebar.number_input(f"% {asset}", value=DEFAULT_MER[TICKERS_DICT[asset]]*100, step=0.01, key=f"fee_in_{asset}") / 100

st.header("📊 Bornes de l'Allocation")
asset_bounds = {}
cols = st.columns(4)
for i, asset in enumerate(TICKERS_DICT.keys()):
    with cols[i % 4]:
        st.write(f"**{asset}**")
        b_min = st.number_input("Min %", 0, 100, 0, key=f"min_f_{asset}") / 100
        # ON FORCE LE MAX CASH A 0 DANS L'UI
        default_val = 0 if "Cash" in asset else 40
        b_max = st.number_input("Max %", 0, 100, default_val, key=f"max_f_{asset}") / 100
        asset_bounds[asset] = (b_min, b_max)

try:
    data = get_market_data(TICKERS_DICT)
    rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
    borrow_cost = rfr + (spread_bps / 10000)

    # Calcul des rendements nets
    fees_series = pd.Series({a: st.session_state[f"fee_in_{a}"]/100 for a in TICKERS_DICT.keys()})
    if mode_cma == "Manuel":
        exp_rets = pd.Series({a: st.session_state[f"r_in_{a}"]/100 for a in TICKERS_DICT.keys()}) - fees_series
        v_diag = np.diag([st.session_state[f"v_in_{a}"]/100 for a in TICKERS_DICT.keys()])
        cov_base = pd.DataFrame(v_diag @ data.corr().values @ v_diag, index=data.columns, columns=data.columns)
    else:
        exp_rets = (data.mean() * 12) - fees_series
        cov_base = data.cov() * 12

    # Délissage
    for asset in ILLIQUID_ASSETS:
        cov_base.loc[asset, :] *= (1/alpha)
        cov_base.loc[:, asset] *= (1/alpha)

    w_opt = optimize_portfolio(exp_rets, cov_base, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

    if w_opt is not None:
        # Nettoyage des petites valeurs
        w_opt = np.array([x if x > 0.0001 else 0 for x in w_opt])
        
        # RESULTATS
        st.divider()
        res_cols = st.columns(4)
        port_ret = (w_opt @ exp_rets) - ((np.sum(w_opt)-1)*borrow_cost)
        port_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
        
        res_cols[1].metric("Rendement Net", f"{port_ret:.2%}")
        res_cols[2].metric("Volatilité", f"{port_vol:.2%}")
        res_cols[3].metric("Frais Totaux", f"{np.sum(w_opt * fees_series):.2%}")

        # GRAPHIQUES
        g_cola, g_colb = st.columns(2)
        with g_cola:
            df_pie = pd.DataFrame({"Actif": TICKERS_DICT.keys(), "Poids": w_opt})
            df_pie = df_pie[df_pie["Poids"] > 0]
            st.plotly_chart(px.pie(df_pie, values="Poids", names="Actif", title="Répartition réelle", hole=0.4), use_container_width=True)
        with g_colb:
            st.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=w_opt*100, title="Poids par classe d'actif (%)"), use_container_width=True)
            
        # Export
        st.download_button("📥 Télécharger CSV", df_pie.to_csv(index=False).encode('utf-8'), "mon_portefeuille.csv")
    else:
        st.error("⚠️ L'optimiseur ne trouve pas de solution avec un Max Cash à 0%. Essayez de monter le levier ou de baisser la cible de rendement.")

except Exception as e:
    st.error(f"Erreur : {e}")
