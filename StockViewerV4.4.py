import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Page config
st.set_page_config(page_title="StockViewer", layout="wide")

# Custom CSS voor oranje titel
st.markdown("""
    <style>
    .main-title {
        color: #ff8c00;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Titel
st.markdown('<h1 class="main-title">StockViewer</h1>', unsafe_allow_html=True)

# Sidebar voor inputs
with st.sidebar:
    st.header("Instellingen")
    ticker = st.text_input("Ticker", value="AAPL", placeholder="Bijv. AAPL, MSFT, GOOG")
    period = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)

# Helper functies
def safe_get(info, key, default=None):
    """Veilig waarde ophalen uit info dict, retourneert default als None of niet gevonden"""
    value = info.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value

def format_currency(value, default="Niet beschikbaar"):
    """Formatteer waarde als currency"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if abs(value) >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"

def format_number(value, decimals=2, default="Niet beschikbaar"):
    """Formatteer nummer"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return f"{value:,.{decimals}f}"

def format_percentage(value, default="Niet beschikbaar"):
    """Formatteer als percentage"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return f"{value*100:.2f}%"

# =============================
# Excel-style waarderingsmodel (Bear / Base / Bull)
# =============================
@st.cache_data(ttl=60 * 30)
def fetch_yahoo_bundle(ticker_symbol: str):
    """Haal alleen serializable data op (geen yfinance Ticker object in cache)."""
    t = yf.Ticker(ticker_symbol)
    info_local = t.info or {}
    income_stmt = t.income_stmt  # annual income statement
    hist_local = t.history(period="5d", interval="1d")
    return {
        "info": info_local,
        "income_stmt": income_stmt,
        "hist": hist_local,
    }

def pick_latest_value(df: pd.DataFrame, possible_rows: list[str]):
    """Pak de meest recente waarde uit income statement voor een van de mogelijke rij-namen."""
    if df is None or df.empty:
        return None
    for r in possible_rows:
        if r in df.index:
            cols = list(df.columns)
            if not cols:
                return None
            # Kies meest recente kolom (meestal nieuwste datum)
            try:
                latest_col = max(cols)
            except Exception:
                latest_col = cols[0]
            val = df.loc[r, latest_col]
            try:
                return float(val)
            except Exception:
                return None
    return None

def compute_scenarios(
    market_cap_now: float,
    price_now: float,
    shares_out: float,
    revenue_now: float,
    years: int,
    scenarios: dict
):
    """Excel-logica: Omzet -> Netto winst (marge) -> Market cap (P/E) -> Rendement/CAGR."""
    rows = []
    for name, s in scenarios.items():
        rev_cagr = float(s["rev_cagr"])
        net_margin = float(s["net_margin"])
        pe = float(s["pe"])

        revenue_future = revenue_now * (1 + rev_cagr) ** years
        net_income_future = revenue_future * net_margin
        market_cap_future = net_income_future * pe

        total_return = (market_cap_future / market_cap_now) - 1 if market_cap_now else None
        cagr_return = (market_cap_future / market_cap_now) ** (1 / years) - 1 if market_cap_now else None

        price_target = (market_cap_future / shares_out) if shares_out else None

        rows.append({
            "Scenario": name,
            "Omzet (nu)": revenue_now,
            f"Omzet ({years}j)": revenue_future,
            f"Netto winst ({years}j)": net_income_future,
            f"Market cap ({years}j)": market_cap_future,
            "Koers (nu)": price_now,
            f"Koersdoel ({years}j)": price_target,
            "Totaal rendement": total_return,
            "CAGR": cagr_return,
            "Aanname: omzet CAGR": rev_cagr,
            "Aanname: netto marge": net_margin,
            "Aanname: P/E": pe,
        })

    return pd.DataFrame(rows).set_index("Scenario")

def format_scenario_df(df_in: pd.DataFrame):
    """Maak de scenario-tabel netjes leesbaar voor Streamlit."""
    df = df_in.copy()

    money_cols = [c for c in df.columns if ("Omzet" in c) or ("winst" in c) or ("Market cap" in c)]
    price_cols = [c for c in df.columns if ("Koers" in c)]
    pct_cols = [c for c in df.columns if c in ["Totaal rendement", "CAGR"]]

    for c in money_cols:
        df[c] = df[c].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and pd.notna(x) else "—")
    for c in price_cols:
        df[c] = df[c].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and pd.notna(x) else "—")
    for c in pct_cols:
        df[c] = df[c].apply(lambda x: f"{x*100:,.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "—")

    if "Aanname: omzet CAGR" in df.columns:
        df["Aanname: omzet CAGR"] = df["Aanname: omzet CAGR"].apply(lambda x: f"{x*100:,.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "—")
    if "Aanname: netto marge" in df.columns:
        df["Aanname: netto marge"] = df["Aanname: netto marge"].apply(lambda x: f"{x*100:,.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "—")
    if "Aanname: P/E" in df.columns:
        df["Aanname: P/E"] = df["Aanname: P/E"].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) and pd.notna(x) else "—")

    return df

# Haal data op
if ticker:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Haal historische data op
        hist = stock.history(period=period)
        
        if hist.empty:
            st.error(f"Geen data gevonden voor {ticker}. Controleer of de ticker correct is.")
            st.stop()
        
        # Haal dividend data op
        dividends = stock.dividends
        
        # Huidige koers en verandering
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Koers", "Dividend", "Waardering", "Voorspelling"])
        
        # TAB 1: Koers
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Huidige Koers",
                    f"${current_price:.2f}",
                    delta=f"${price_change:.2f} ({price_change_pct:+.2f}%)",
                    delta_color="normal" if price_change >= 0 else "inverse"
                )
            
            with col2:
                st.metric(
                    "Aantal uitstaande aandelen",
                    format_number(hist['Volume'].iloc[-1], decimals=0)
                )
            
            # Lijnchart
            st.subheader("Koersontwikkeling")
            st.line_chart(hist['Close'])
        
        # TAB 2: Dividend
        with tab2:
            if not dividends.empty:
                # Filter dividend op dezelfde periode als koers
                start_date = hist.index[0]
                end_date = hist.index[-1]
                filtered_dividends = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)]
                
                # Haal volledige dividend geschiedenis op (5 jaar) voor analyse
                hist_5y = stock.history(period="5y")
                if not hist_5y.empty:
                    dividends_5y = dividends[(dividends.index >= hist_5y.index[0]) & (dividends.index <= hist_5y.index[-1])]
                else:
                    dividends_5y = dividends
                
                if not filtered_dividends.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_dividend = filtered_dividends.sum()
                        st.metric("Totaal Dividend (periode)", f"${total_dividend:.2f}")
                    
                    with col2:
                        avg_dividend = filtered_dividends.mean()
                        st.metric("Gemiddeld Dividend", f"${avg_dividend:.4f}")
                    
                    with col3:
                        dividend_yield_val = safe_get(info, "dividendYield")
                        if dividend_yield_val:
                            st.metric("Dividend Yield", format_percentage(dividend_yield_val))
                        else:
                            st.metric("Dividend Yield", "Niet beschikbaar")
                    
                    with col4:
                        payout_ratio = safe_get(info, "payoutRatio")
                        if payout_ratio:
                            st.metric("Payout Ratio", format_percentage(payout_ratio))
                        else:
                            st.metric("Payout Ratio", "Niet beschikbaar")
                    
                    st.markdown("---")
                    
                    # Dividend geschiedenis grafiek
                    st.subheader("Dividend Historie (Gekozen Periode)")
                    st.bar_chart(filtered_dividends)
                    
                    # Uitgebreide analyse
                    if len(dividends_5y) > 1:
                        st.markdown("---")
                        st.subheader("Dividend Analyse (5 Jaar)")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Groei analyse
                            dividend_yearly = dividends_5y.groupby(dividends_5y.index.year).sum()
                            if len(dividend_yearly) > 1:
                                st.markdown("### Jaarlijks Totaal Dividend")
                                st.bar_chart(dividend_yearly)
                                
                                # Bereken groei
                                if len(dividend_yearly) >= 2:
                                    growth_rates = []
                                    for i in range(1, len(dividend_yearly)):
                                        prev = dividend_yearly.iloc[i-1]
                                        curr = dividend_yearly.iloc[i]
                                        if prev > 0:
                                            growth = ((curr - prev) / prev) * 100
                                            growth_rates.append(growth)
                                    
                                    if growth_rates:
                                        avg_growth = np.mean(growth_rates)
                                        st.metric("Gemiddelde Jaarlijkse Groei", f"{avg_growth:.2f}%")
                        
                        with col_b:
                            # Dividend per share trend
                            st.markdown("### Dividend Per Share Trend")
                            if len(dividends_5y) > 4:
                                # Bereken rolling average
                                rolling_avg = dividends_5y.rolling(window=4).mean()
                                df_trend = pd.DataFrame({
                                    'Dividend': dividends_5y,
                                    'Rolling Average (4x)': rolling_avg
                                })
                                st.line_chart(df_trend)
                            
                            # Laatste dividend info
                            st.markdown("### Recent Dividend")
                            last_dividend = dividends_5y.iloc[-1]
                            last_date = dividends_5y.index[-1]
                            st.write(f"**Laatste uitbetaling:** {last_date.strftime('%d-%m-%Y')}")
                            st.write(f"**Bedrag:** ${last_dividend:.4f}")
                            
                            # Voorspelling volgende dividend
                            if len(dividends_5y) >= 4:
                                recent_avg = dividends_5y.tail(4).mean()
                                st.write(f"**Gemiddeld (laatste 4x):** ${recent_avg:.4f}")
                                
                                # Schat volgende dividend datum (meestal kwartaal of halfjaar)
                                if len(dividends_5y) >= 2:
                                    intervals = []
                                    for i in range(1, min(5, len(dividends_5y))):
                                        days_diff = (dividends_5y.index[-i] - dividends_5y.index[-i-1]).days
                                        intervals.append(days_diff)
                                    
                                    if intervals:
                                        avg_interval = np.mean(intervals)
                                        next_date_estimate = last_date + pd.Timedelta(days=avg_interval)
                                        st.write(f"**Geschatte volgende uitbetaling:** {next_date_estimate.strftime('%d-%m-%Y')}")
                    
                    # Dividend details tabel
                    st.markdown("---")
                    st.subheader("Dividend Details")
                    df_dividends_display = filtered_dividends.to_frame('Dividend')
                    df_dividends_display.index.name = 'Datum'
                    df_dividends_display = df_dividends_display.sort_index(ascending=False)
                    st.dataframe(df_dividends_display, use_container_width=True)
                    
                else:
                    st.info(f"Geen dividend uitbetaald in de gekozen periode ({period})")
            else:
                st.info(f"{ticker} betaalt geen dividend uit")
                st.markdown("""
                **Wat betekent dit?**
                - Het bedrijf keert momenteel geen dividend uit aan aandeelhouders
                - Dit kan betekenen dat het bedrijf winst herinvesteert in groei
                - Veel tech bedrijven betalen geen dividend maar focussen op groei
                """)
        
        # TAB 3: Waardering
        with tab3:
            st.subheader("Fundamentals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Bedrijfsgegevens")
                market_cap = safe_get(info, "marketCap")
                revenue = safe_get(info, "totalRevenue")
                net_income = safe_get(info, "netIncomeToCommon")
                shares_outstanding = safe_get(info, "sharesOutstanding")
                
                st.write(f"**Market Cap:** {format_currency(market_cap)}")
                st.write(f"**Revenue:** {format_currency(revenue)}")
                st.write(f"**Net Income:** {format_currency(net_income)}")
                st.write(f"**Shares Outstanding:** {format_number(shares_outstanding, decimals=0) if shares_outstanding else 'Niet beschikbaar'}")
            
            with col2:
                st.markdown("### Financiële Positie")
                cash = safe_get(info, "totalCash")
                debt = safe_get(info, "totalDebt")
                
                # Bereken Enterprise Value
                ev = None
                if market_cap and debt is not None and cash is not None:
                    ev = market_cap + debt - cash
                
                st.write(f"**Cash:** {format_currency(cash)}")
                st.write(f"**Debt:** {format_currency(debt)}")
                st.write(f"**Enterprise Value:** {format_currency(ev)}")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### Ratio's")
                eps = safe_get(info, "trailingEps")
                pe = safe_get(info, "trailingPE")
                price_to_sales = safe_get(info, "priceToSalesTrailing12Months")
                dividend_yield = safe_get(info, "dividendYield")
                
                st.write(f"**EPS:** {format_currency(eps) if eps else 'Niet beschikbaar'}")
                st.write(f"**P/E Ratio:** {format_number(pe, decimals=2) if pe else 'Niet beschikbaar'}")
                st.write(f"**Price-to-Sales:** {format_number(price_to_sales, decimals=2) if price_to_sales else 'Niet beschikbaar'}")
                st.write(f"**Dividend Yield:** {format_percentage(dividend_yield) if dividend_yield else 'Niet beschikbaar'}")
            
            with col4:
                st.markdown("### Uitleg")
                st.markdown("""
                **Market Cap:** Totale marktwaarde van het bedrijf (aantal aandelen × koers)
                
                **Revenue:** Totale omzet van het bedrijf
                
                **Net Income:** Winst na alle kosten
                
                **Enterprise Value:** Marktwaarde + schulden - cash (totale bedrijfswaarde)
                
                **EPS:** Winst per aandeel
                
                **P/E Ratio:** Koers gedeeld door winst per aandeel (hoe duur het aandeel is)
                
                **Price-to-Sales:** Koers gedeeld door omzet per aandeel
                
                **Dividend Yield:** Jaarlijks dividend als percentage van de koers
                """)
        
        # TAB 4: Voorspelling
        with tab4:
            st.subheader("Koers voorspelling")
            st.caption("Grove schatting op basis van omzetgroei, winstmarge en P/E-multiple (data: Yahoo Finance).")

            with st.expander("Hoe maakt deze bot de voorspelling?"):
                st.markdown(
                    """
Deze voorspelling is **geen exacte koersvoorspelling**, maar een **waarderingsschatting** met een bandbreedte.

**Stap 1 — Omzet projecteren (groei)**  
De bot pakt een omzetgroei uit Yahoo Finance (`revenueGrowth`). Als die ontbreekt gebruikt hij een conservatieve default.

**Stap 2 — Winst schatten (marge)**  
De bot gebruikt de winstmarge uit Yahoo Finance (`profitMargins`). Als die ontbreekt, schat hij marge via `net income / omzet` of een default.

**Stap 3 — Waardering berekenen (P/E)**  
De bot kiest een P/E-multiple uit Yahoo Finance (`forwardPE`, anders `trailingPE`). Daarmee rekent hij een toekomstige market cap:
- `Omzet_future = Omzet_nu × (1 + CAGR)^jaren`
- `Winst_future = Omzet_future × marge`
- `MarketCap_future = Winst_future × P/E`

**Stap 4 — Koersdoel + rendement**  
Als het aantal aandelen bekend is, wordt daar een koersdoel van gemaakt:
- `Koersdoel = MarketCap_future / shares outstanding`

En daarna berekent hij ook **totaal rendement** en **CAGR** t.o.v. de market cap van nu.

**Bandbreedte**  
Je ziet drie uitkomsten: **Conservatief / Basis / Optimistisch (Bear, Base, Bull)** (de bot maakt die door de basis-aannames iets lager/hoger te zetten).
                    """
                )

            bundle = fetch_yahoo_bundle(ticker)
            info_bundle = bundle.get("info", {})
            income_stmt = bundle.get("income_stmt")
            hist_small = bundle.get("hist")

            # Huidige koers (laatste close)
            price_now = None
            if isinstance(hist_small, pd.DataFrame) and (not hist_small.empty) and ("Close" in hist_small.columns):
                try:
                    price_now = float(hist_small["Close"].iloc[-1])
                except Exception:
                    price_now = None

            # Kern inputs uit Yahoo (met fallbacks)
            market_cap_now = safe_get(info_bundle, "marketCap")
            shares_out = safe_get(info_bundle, "sharesOutstanding")

            revenue_now = pick_latest_value(income_stmt, ["Total Revenue", "TotalRevenue"])
            if revenue_now is None:
                revenue_now = safe_get(info_bundle, "totalRevenue")

            # Suggestie voor marge (profitMargins of netIncome/revenue)
            suggested_margin = safe_get(info_bundle, "profitMargins")
            if suggested_margin is None:
                ni = pick_latest_value(income_stmt, [
                    "Net Income",
                    "NetIncome",
                    "Net Income Common Stockholders",
                    "NetIncomeCommonStockholders"
                ])
                if ni is not None and revenue_now:
                    try:
                        suggested_margin = float(ni) / float(revenue_now)
                    except Exception:
                        suggested_margin = None

            # Data check
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Koers (laatste close)", f"${price_now:,.2f}" if price_now else "—")
            with c2:
                st.metric("Market cap (nu)", format_currency(market_cap_now))
            with c3:
                st.metric("Shares outstanding", format_number(shares_out, decimals=0) if shares_out else "Niet beschikbaar")
            with c4:
                st.metric("Omzet (laatste jaar)", format_currency(revenue_now))

            with st.expander("Handmatige overrides (alleen invullen als Yahoo data mist)"):
                market_cap_now = st.number_input(
                    "Market cap nu (USD)",
                    value=float(market_cap_now) if market_cap_now else 0.0,
                    min_value=0.0,
                    step=1.0,
                ) or market_cap_now

                shares_out = st.number_input(
                    "Shares outstanding",
                    value=float(shares_out) if shares_out else 0.0,
                    min_value=0.0,
                    step=1.0,
                ) or shares_out

                revenue_now = st.number_input(
                    "Omzet laatste jaar (USD)",
                    value=float(revenue_now) if revenue_now else 0.0,
                    min_value=0.0,
                    step=1.0,
                ) or revenue_now

            if not market_cap_now or not revenue_now:
                st.error("Voor deze waardering zijn minimaal **Market cap nu** en **Omzet** nodig. Vul ontbrekende waarden in bij de overrides.")
                st.stop()

            years = st.slider("Horizon (jaren)", min_value=1, max_value=10, value=5)

            # Simpele (automatische) aannames — geen sliders voor gebruikers
            # We pakken logische defaults uit Yahoo Finance en maken daar een grove schatting van.
            # - Groei (CAGR): revenueGrowth als die er is, anders een conservatieve default
            # - Marge: profitMargins (of netIncome/revenue), anders default
            # - P/E: forwardPE (anders trailingPE), anders default

            # 1) Omzetgroei (CAGR)
            rev_growth = safe_get(info_bundle, "revenueGrowth")
            if rev_growth is None:
                rev_cagr_base = 0.10
            else:
                # clamp zodat extreme waarden niet het model slopen
                rev_cagr_base = float(max(-0.20, min(0.40, rev_growth)))

            # 2) Netto marge
            net_margin_base = suggested_margin
            if net_margin_base is None:
                net_margin_base = 0.10
            else:
                net_margin_base = float(max(-0.10, min(0.40, net_margin_base)))

            # 3) P/E multiple
            pe_base = safe_get(info_bundle, "forwardPE")
            if pe_base is None:
                pe_base = safe_get(info_bundle, "trailingPE")
            if pe_base is None:
                pe_base = 20.0
            else:
                pe_base = float(max(5.0, min(60.0, pe_base)))

            # We tonen een eenvoudige bandbreedte (conservatief / basis / optimistisch)
            scenarios = {
                "Conservatief": {
                    "rev_cagr": max(-0.20, rev_cagr_base * 0.7),
                    "net_margin": max(-0.10, net_margin_base * 0.8),
                    "pe": max(5.0, pe_base * 0.8),
                },
                "Basis": {
                    "rev_cagr": rev_cagr_base,
                    "net_margin": net_margin_base,
                    "pe": pe_base,
                },
                "Optimistisch": {
                    "rev_cagr": min(0.40, rev_cagr_base * 1.3),
                    "net_margin": min(0.40, net_margin_base * 1.2),
                    "pe": min(60.0, pe_base * 1.2),
                },
            }

            st.markdown("### Aannames (automatisch)")
            st.caption("Deze schatting gebruikt data van Yahoo Finance. Je hoeft niets in te vullen; het model maakt automatisch een realistische bandbreedte.")

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Omzet CAGR (basis)", f"{rev_cagr_base*100:.1f}%")
            with a2:
                st.metric("Netto marge (basis)", f"{net_margin_base*100:.1f}%")
            with a3:
                st.metric("P/E (basis)", f"{pe_base:.1f}")

            with st.expander("Geavanceerd (optioneel): aannames aanpassen"):
                st.caption("Alleen als je wilt tweaken. Laat anders zo voor een snelle schatting.")
                rev_cagr_base = st.slider("Omzet CAGR (basis)", -0.20, 0.40, float(rev_cagr_base), 0.01)
                net_margin_base = st.slider("Netto marge (basis)", -0.10, 0.40, float(net_margin_base), 0.01)
                pe_base = st.slider("P/E (basis)", 5.0, 60.0, float(pe_base), 0.5)

                scenarios = {
                    "Bear": {
                        "rev_cagr": max(-0.20, rev_cagr_base * 0.7),
                        "net_margin": max(-0.10, net_margin_base * 0.8),
                        "pe": max(5.0, pe_base * 0.8),
                    },
                    "Base": {
                        "rev_cagr": rev_cagr_base,
                        "net_margin": net_margin_base,
                        "pe": pe_base,
                    },
                    "Bull": {
                        "rev_cagr": min(0.40, rev_cagr_base * 1.3),
                        "net_margin": min(0.40, net_margin_base * 1.2),
                        "pe": min(60.0, pe_base * 1.2),
                    },
                }

            df_raw = compute_scenarios(
                market_cap_now=float(market_cap_now),
                price_now=price_now,
                shares_out=float(shares_out) if shares_out else None,
                revenue_now=float(revenue_now),
                years=years,
                scenarios=scenarios,
            )

            st.markdown("---")
            st.subheader("Verwachtingstabel")
            st.dataframe(format_scenario_df(df_raw), use_container_width=True)

            st.markdown("---")
            st.subheader("Snelle samenvatting")
            if df_raw["CAGR"].notna().any():
                best = df_raw["CAGR"].idxmax()
                worst = df_raw["CAGR"].idxmin()
                s1, s2 = st.columns(2)
                with s1:
                    st.info(f"Beste scenario op CAGR: **{best}** ({df_raw.loc[best, 'CAGR']*100:.1f}% p/j)")
                with s2:
                    st.warning(f"Slechtste scenario op CAGR: **{worst}** ({df_raw.loc[worst, 'CAGR']*100:.1f}% p/j)")
            else:
                st.info("Niet genoeg data om CAGR te tonen (controleer inputs/overrides).")

            with st.expander("Debug (Yahoo keys)"):
                st.write({
                    "shortName": info_bundle.get("shortName"),
                    "currency": info_bundle.get("currency"),
                    "marketCap": info_bundle.get("marketCap"),
                    "sharesOutstanding": info_bundle.get("sharesOutstanding"),
                    "totalRevenue": info_bundle.get("totalRevenue"),
                    "profitMargins": info_bundle.get("profitMargins"),
                    "trailingPE": info_bundle.get("trailingPE"),
                    "forwardPE": info_bundle.get("forwardPE"),
                })

    except Exception as e:
        st.error("Er ging iets mis tijdens het ophalen of verwerken van de data.")
        st.exception(e)
