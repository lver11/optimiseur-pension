import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Stratégie Pension Canada", layout="wide")

# --- 1. CONFIGURATION ---
# L'ordre ici est l'ordre de référence pour tout le programme
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
    # On télécharge et on s'assure que l'ordre des colonnes respecte exactement TICKERS_DICT
    assets_keys = list(tickers.keys())
    tickers_list = [tickers[k] for k in assets_keys]
    raw_data = yf.download(tickers_list, period="10y", interval="1mo")
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
    
    # Renommer les colonnes par leurs noms lisibles
    inv_map = {v: k for k, v in tickers.items()}
    data = data.rename(columns=inv_map)
    
    # Réorganiser les colonnes pour être CERTAIN de l'ordre
    data = data[assets_keys]
    return data.pct_change().dropna()

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
    
    constraints = [
        cp.sum(w) <= lev_limit, 
        cp.sum(w) >= 1.0, 
        net_return >= target_ret
    ]
    
    # Application rigoureuse des bornes (l'ordre des index doit correspondre)
    for i, name in enumerate(returns_series.index):
        constraints.append(w[i] >= asset_bounds[name][0])
        constraints.append(w[i] <= asset_bounds[name][1])
        
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in illiquid_list]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    try:
        prob.solve(solver=cp.OSQP)
    except:
        prob.solve()
    return w.value if w.value is not None else None

# --- 2. INTERFACE ---
st.title("🏛️ Station de Recherche : Portefeuille Institutionnel")
tab_main, tab_corr, tab_arch = st.tabs(["📊 Optimisation", "📈 Corrélations", "🔍 Architecture et Frais"])

with st.sidebar:
    st.header("⚙️ Configuration")
    lev_max = st.slider("Levier Brut Max", 1.0, 2.0, 1.25)
    target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
    alpha = st.slider("Alpha (Délissage)", 0.3, 1.0, 0.5)
    max_i = st.slider("Max Alternatifs (%)", 10, 80, 45) / 100
    spread_bps = st.number_input("Spread Levier (bps)", value=120)
    
    st.header("🔮 Anticipations (CMA)")
    mode_cma = st.radio("Source CMA :", ["Historique", "Manuel"])
    user_rets, user_vols = {}, {}
    
    # On parcourt les clés pour garantir l'ordre
    for asset in TICKERS_DICT.keys():
        if mode_cma == "Manuel":
            user_rets[asset] = st.number_input(f"Rend. {asset} %", value=7.0, step=0.1, key=f"r_{asset}") / 100
            user_vols[asset] = st.number_input(f"Vol. {asset} %", value=12.0, step=0.1, key=f"v_{asset}") / 100
        
        # Frais
        st.sidebar.markdown(f"---")
        custom_fee = st.sidebar.number_input(f"Frais {asset} (%)", value=DEFAULT_MER[TICKERS_DICT[asset]]*100, step=0.05, key=f"fee_v2_{asset}") / 100
        user_rets[asset if mode_cma == "Manuel" else asset] = custom_fee # Juste pour stockage temporaire

with tab_main:
    st.header("📊 Politique de Placement (Bornes)")
    asset_bounds = {}
    cols = st.columns(4)
    for i, asset in enumerate(TICKERS_DICT.keys()):
        with cols[i % 4]:
            b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"nmin_v2_{asset}") / 100
            
            # FORCE RESET DU CASH A 0%
            if "Cash" in asset:
                def_max = 0
            elif "Mondiales" in asset:
                def_max = 80
            else:
                def_max = 40
                
            b_max = st.number_input(f"Max {asset} %", 0, 100, def_max, key=f"nmax_v2_{asset}") / 100
            asset_bounds[asset] = (b_min, b_max)

    try:
        hist_data = get_market_data(TICKERS_DICT)
        rfr = (1 + hist_data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)

        # Extraction des rendements bruts selon le mode
        if mode_cma == "Manuel":
            exp_raw = pd.Series({k: v for k, v in user_rets.items() if k in TICKERS_DICT}) # Correction ici
            # Note: il faut reconstruire exp_raw car user_rets a été utilisé pour les frais plus haut
            exp_raw = pd.Series([st.session_state[f"r_{a}"]/100 for a in TICKERS_DICT.keys()], index=TICKERS_DICT.keys())
        else:
            exp_raw = hist_data.mean() * 12

        # Application des frais (ceux saisis dans la sidebar)
        applied_fees = pd.Series([st.session_state[f"fee_v2_{a}"]/100 for a in TICKERS_DICT.keys()], index=TICKERS_DICT.keys())
        exp_rets = exp_raw - applied_fees
        
        # Matrice de Covariance
        if mode_cma == "Manuel":
            vols_manual = [st.session_state[f"v_{a}"]/100 for a in TICKERS_DICT.keys()]
            v_diag = np.diag(vols_manual)
            adj_cov_base = pd.DataFrame(v_diag @ hist_data.corr().values @ v_diag, index=hist_data.columns, columns=hist_data.columns)
        else:
            adj_cov_base = hist_data.cov() * 12

        adj_cov = desmooth_cov(adj_cov_base, alpha, ILLIQUID_ASSETS)
        w_opt = optimize_portfolio(exp_rets, adj_cov, lev_max, target_r, borrow_cost, asset_bounds, max_i, ILLIQUID_ASSETS)

        if w_opt is not None:
            # Pour l'affichage, on arrondit les valeurs proches de zero
            w_opt[w_opt < 1e-4] = 0
            
            port_ret_net = (w_opt @ exp_rets) - ((np.sum(w_opt)-1) * borrow_cost)
            port_vol = np.sqrt(w_opt.T @ adj_cov @ w_opt)
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Net", f"{port_ret_net:.2%}")
            m2.metric("Volatilité", f"{port_vol:.2%}")
            m3.metric("Frais Totaux", f"{np.sum(w_opt * applied_fees):.2%}")
            m4.metric("Carry Levier", f"{(w_opt @ exp_rets / np.sum(w_opt)) - borrow_cost:+.2%}")

            cola, colb = st.columns(2)
            with cola: 
                # On filtre les poids à 0 pour un graphique propre
                mask = w_opt > 0
                st.plotly_chart(px.pie(values=w_opt[mask], names=hist_data.columns[mask], title="Allocation du Capital", hole=0.4), use_container_width=True)
            with colb: 
                st.plotly_chart(px.bar(x=hist_data.columns, y=w_opt*100, title="Poids par Actif (%)"), use_container_width=True)

            # Export
            export_df = pd.DataFrame({"Actif": hist_data.columns, "Poids %": (w_opt*100).round(2)})
            st.download_button("📥 Export CSV", export_df.to_csv(index=False).encode('utf-8'), "allocation.csv", "text/csv")
        else:
            st.error("⚠️ Pas de solution trouvée. Vos bornes sont peut-être trop restrictives (ex: trop de Cash interdit alors que la cible est basse).")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

# --- TAB CORR ET ARCHIVE (Simplifiés pour le code complet) ---
with tab_corr:
    if 'hist_data' in locals():
        st.plotly_chart(px.imshow(hist_data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
