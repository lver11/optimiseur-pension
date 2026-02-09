import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="Terminal CIO - Master Control", layout="wide")

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

ILLIQUID_ASSETS = ["Infrastructures", "Dette Privée (Proxy)", "Hypothèques Comm. (Proxy)", "Immobilier Listé"]

@st.cache_data
def get_market_data(tickers):
    keys = list(tickers.keys())
    raw = yf.download([tickers[k] for k in keys], period="10y", interval="1mo")
    df = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df[keys].pct_change().dropna()

def optimize_portfolio(returns_series, cov_matrix, lev_limit, target_ret, borrow_cost, asset_bounds, max_ill_limit, pe_fixed_weight):
    n = len(returns_series)
    w = cp.Variable(n)
    lev_amt = cp.sum(w) - 1
    net_return = w @ returns_series.values - (lev_amt * borrow_cost)
    
    # Sécurité PSD pour éviter les erreurs DCP
    S = (cov_matrix.values + cov_matrix.values.T) / 2 + np.eye(n) * 1e-4
    risk = cp.quad_form(w, cp.psd_wrap(S))
    
    constraints = [
        cp.sum(w) <= lev_limit, 
        cp.sum(w) >= 1.0, 
        net_return >= target_ret, 
        w >= 0
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
    prob.solve(solver=cp.ECOS)
    return w.value if w.value is not None else None

# --- 2. INTERFACE ET LOGIQUE ---
try:
    data = get_market_data(TICKERS_DICT)

    with st.sidebar:
        st.header("⚙️ Gouvernance CIO")
        lev_max = st.slider("Levier Brut Max", 1.0, 2.5, 1.25)
        target_r = st.slider("Cible Rendement NET (%)", 4.0, 10.0, 6.5) / 100
        
        st.subheader("📌 Allocation Stratégique")
        # CASE DE FIXATION MANUELLE DU PE
        pe_fix = st.number_input("Fixer Placement Privé %", 0.0, 30.0, 10.0) / 100
        
        alpha = st.slider("Alpha (Délissage)", 0.2, 1.0, 0.4)
        spread_bps = st.number_input("Spread Levier (bps)", value=94)

    tab_opt, tab_risk = st.tabs(["📊 Optimisation", "📈 Analyse Risque"])

    with tab_opt:
        asset_bounds = {}
        st.write("### Bornes Tactiques (Actifs Liquides)")
        cols = st.columns(4)
        for i, asset in enumerate(TICKERS_DICT.keys()):
            if asset == "Placement Privé (PE)": continue
            with cols[i % 4]:
                b_min = st.number_input(f"Min {asset} %", 0, 100, 0, key=f"min_{asset}")/100
                d_max = 0 if "Cash" in asset else 50
                b_max = st.number_input(f"Max {asset} %", 0, 100, d_max, key=f"max_{asset}")/100
                asset_bounds[asset] = (b_min, b_max)

        rfr = (1 + data["Cash (RFR)"].mean())**12 - 1
        borrow_cost = rfr + (spread_bps / 10000)
        
        # CMA
        exp_raw = data.mean()*12
        cov_base = data.cov()*12
        # Délissage
        for a in ILLIQUID_ASSETS + ["Placement Privé (PE)"]:
            cov_base.loc[a, :], cov_base.loc[:, a] = cov_base.loc[a, :] * (1/alpha), cov_base.loc[:, a] * (1/alpha)

        applied_fees = pd.Series({a: 0.0025 for a in TICKERS_DICT.keys()}) 

        w_opt = optimize_portfolio(exp_raw - applied_fees, cov_base, lev_max, target_r, borrow_cost, asset_bounds, 0.5, pe_fix)
        
        if w_opt is not None:
            w_opt = np.array([x if x > 0.0001 else 0 for x in w_opt])
            p_vol = np.sqrt(w_opt.T @ cov_base @ w_opt)
            p_ret = (w_opt @ (exp_raw - applied_fees)) - ((np.sum(w_opt)-1) * borrow_cost if np.sum(w_opt) > 1 else 0)
            
            st.divider()
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Expected Net Return", f"{p_ret:.2%}")
            res_c2.metric("Portfolio Vol", f"{p_vol:.2%}")
            res_c3.metric("Leverage Used", f"{np.sum(w_opt):.2f}x")

            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(pd.DataFrame({"A": TICKERS_DICT.keys(), "P": w_opt}), values="P", names="A", hole=0.4, title="Mix Actifs"), use_container_width=True)
            rc = (w_opt * (cov_base @ w_opt)) / (p_vol**2 if p_vol > 0 else 1)
            c2.plotly_chart(px.bar(x=list(TICKERS_DICT.keys()), y=rc*100, title="Risk Contribution (%)"), use_container_width=True)
        else:
            st.error("⚠️ Incohérence mathématique : la cible est inatteignable avec ces bornes.")

    with tab_risk:
        st.header("Matrice de Corrélation Historique")
        st.plotly_chart(px.imshow(data.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=700), use_container_width=True)

except Exception as e:
    st.error(f"Erreur technique : {e}")
