import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Contrôle Institutionnel", layout="wide")

# --- 1. CONFIGURATION DES ACTIFS ---
TICKERS_DICT = {
    "Actions US (Unhedged)": "VFV.TO",
    "Actions Mondiales (Unhedged)": "VXC.TO",
    "Placement Privé (PE)": "PSP",
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
    "VFV.TO": 0.0009, "VXC.TO": 0.0021, "PSP": 0.0180, "VEE.TO": 0.0024, 
    "VWOB": 0.0020, "ZGI.TO": 0.0061, "VRE.TO": 0.0039, "DBC": 0.0085, 
    "XSU.TO": 0.0033, "VAB.TO": 0.0009, "XHY.TO": 0.0061, "XCB.TO": 0.0018, "PSA.TO": 0.0015
}

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    keys = list(tickers.keys())
    raw = yf.download([tickers[k] for k in keys], period="10y", interval="1mo")
    df = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df[keys].pct_change().dropna()

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, pe_fixed_weight, vol_cap):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    S = (cov_matrix.values + cov_matrix.values.T) / 2 + np.eye(n) * 1e-4
    risk = cp.quad_form(w, cp.psd_wrap(S))
    constraints = [
        cp.sum(w) <= lev_limit, cp.sum(w) >= 1.0, net_return >= target_ret, 
        w >= 0, risk <= (vol_cap ** 2)
    ]
    for i, name in enumerate(returns_series.index):
        if name == "Placement Privé (PE)":
            constraints.append(w[i] == pe_fixed_weight)
        else:
            constraints.append(w[i] >= asset_bounds[name][0])
            constraints.append(w[i] <= asset_bounds[name][1])
    ill_idx = [i for i, name in enumerate(returns_series.index) if name in ILLIQUID_ASSETS]
    constraints.append(cp.sum(w[ill_idx]) <= max_ill_limit)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    for s in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=s)
            if w.value is not None: break
        except: continue
    return w.value if w.value is not None else None

# --- 2. LOGIQUE ET INTERFACE ---
try:
    data = get_market_data(TICKERS_DICT)

    with st.sidebar:
        st.header("⚙️ Gouvernance CIO")
        lev_max = st.slider("Levier Brut Max", 1.0, 2.5, 1.25)
        target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
        vol_target = st.slider("Cible Volatilité Max (%)", 5.0, 15.0, 10.0) / 100
        pe_fix = st.number_input("Fixer Placement Privé %", 0.0, 30.0, 10.0) / 100
        alpha = st.slider("Alpha (Délissage)", 0.2, 1.0, 0.4)
        spread_bps = st.number_input("Spread Levier (bps)", value=94)
        
        st.header("🔮 Capital Market Assumptions")
        mode_cma = st.radio("Source des données :", ["Historique", "Manuel"])
        user_rets, user_vols = {}, {}
        if mode_cma == "Manuel":
            for asset in TICKERS_DICT.keys():
                st.markdown(f"**{asset}**")
                c1, c2 = st.columns(2)
                user_rets[asset] = c1.number_input(f"Rend. %", 7.0, key=f"r_{asset}")/100
                user_vols[asset] = c2.number_input(f"Vol. %", 12.0, key=f"v_{asset}")/100

    tab_opt, tab_risk, tab_fees, tab_lex = st.tabs(["📊 Optimisation", "📈 Analyse Risque", "💰 Frais", "🔍 Lexique"])

    with tab_lex:
        st.header("📖 Lexique Méthodologique")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### Gestion du Risque
            * **Alpha (Délissage) :** Ajustement de la volatilité pour les actifs évalués périodiquement (PE, Immo). Un Alpha de 0.4 multiplie la volatilité historique par 2.5.
            * **Cible de Volatilité :** Plafond de risque imposé au solveur. Si la cible de rendement exige trop de risque, le modèle réduit le levier.
            * **Placement Privé Fixe :** Ancre mathématique permettant de stabiliser l'optimisation des actifs liquides.
            """)
        with c2:
            st.markdown("""
            ### Ingénierie Financière
            * **Levier Brut :** Exposition totale divisée par le capital propre.
            * **TER (Total Expense Ratio) :** Frais de gestion totaux pondérés par l'allocation réelle (incluant le levier).
            * **Spread Levier :** Prime d'emprunt au-dessus du taux sans risque canadien.
            """)

    with tab_fees:
        st.header("🖊️ Saisie des Frais (MER/RFG)")
        manual_fees = {a: st.number_input(f"Frais {a} (%)", 0.0, 5.0, DEFAULT_MER[TICKERS_DICT[a]]*100, step=0.01, key=f"f_{a}")/100 for a in TICKERS_DICT.keys()}

    with tab_opt:
        asset_bounds = {a: (st.sidebar.number_input(f"Min {a} %", 0, 100, 0, key=f"m_{a}")/100, 0.5) for a in TICKERS_DICT.keys() if a != "Placement Privé (PE)"}
        # Code d'optimisation simplifié pour le rendu
        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        exp_raw = pd.Series(user_rets) if mode_cma == "Manuel" else data.mean()*12
        cov_base = data.cov()*12 # + Délissage
        
        applied_fees_series = pd.Series(manual_fees)
        w_opt = optimize_portfolio(exp_raw - applied_fees_series, cov_base, lev_max, target_r, borrow_cost, {a: (0, 0.5) for a in TICKERS_DICT.keys()}, 0.5, pe_fix, vol_target)
        
        if w_opt is not None:
            w_opt = np.array([x if x > 0.0001 else 0 for x in w_opt])
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            p_ret = (w_opt @ (exp_raw - applied_fees_series)) - ((np.sum(w_opt)-1) * borrow_cost if np.sum(w_opt) > 1 else 0)
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Net Return", f"{p_ret:.2%}")
            m2.metric("Portfolio Vol", f"{p_vol:.2%}")
            m3.metric("Leverage Used", f"{np.sum(w_opt):.2f}x")
            m4.metric("TER Total", f"{np.sum(w_opt * applied_fees_series):.2%}")

            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Asset Mix"), use_container_width=True)
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2 if p_vol > 0 else 1)
            c2.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Risk Contribution (%)"), use_container_width=True)

    with tab_risk:
        st.plotly_chart(px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=700), use_container_width=True)

except Exception as e:
    st.error(f"Erreur : {e}")
