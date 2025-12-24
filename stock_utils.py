import pandas as pd
import yfinance as yf
import numpy as np
from yahooquery import Ticker

# technical and fundamental cals
def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain[:period+1].mean()
    avg_loss = loss[:period+1].mean()

    rsi_values = []
    for i in range(period+1, len(series)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return pd.Series([None]*(period+1) + rsi_values, index=series.index)

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Returns MACD line, Signal line, Histogram.
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

# Sector margin assumptions (adjust as needed)
sector_margin_map = {
    "Technology": 0.20,
    "Communication Services": 0.18,
    "Consumer Cyclical": 0.08,
    "Consumer Defensive": 0.06,
    "Healthcare": 0.12,
    "Financial Services": 0.10,
    "Industrials": 0.08,
    "Energy": 0.07,
    "Utilities": 0.08,
    "Real Estate": 0.05,
    "Basic Materials": 0.07
}

def compute_evrev_valuation(ev_to_rev, next_year_revenue, shares_outstanding, sector):
    """
    Compute valuation for negative EPS companies using EV/Revenue and margin assumptions.
    Returns per-share valuation.
    """
    if not ev_to_rev or not next_year_revenue or not shares_outstanding:
        return None

    margin_assumption = sector_margin_map.get(sector, 0.10)  # default 10% if unknown

    # Implied P/E from EV/Revenue
    implied_pe = ev_to_rev / margin_assumption

    # Target EPS from next-year revenue
    eps_target = (next_year_revenue * margin_assumption) / shares_outstanding

    # Valuation per share
    valuation = eps_target * implied_pe
    return valuation

# Hybrid CAGR: 3 past years + current year + next year
def compute_hybrid_growth(eps_history, eps_curr_year=None, eps_next_year=None, threshold=0.5):
    """
    Hybrid growth calculation:
    - If start EPS > 0: use CAGR.
    - If start EPS <= 0: use % change if avg EPS magnitude > threshold,
      otherwise use delta growth.
    """
    eps_clean = eps_history.dropna().tolist()
    if len(eps_clean) < 3:
        return None
    eps_end = eps_next_year if eps_next_year is not None else eps_curr_year
    if eps_end is None:
        return None

    eps_start = eps_clean[-3]

    # Case 1: Positive start → CAGR
    if eps_start > 0 and eps_end > 0:
        years_spanned = 4
        return (eps_end / eps_start)**(1/years_spanned) - 1

    # Case 2: Negative/zero start → hybrid % vs delta
    eps_series = eps_clean + [eps_end]

    deltas = []
    pct_changes = []
    for i in range(1, len(eps_series)):
        prev, curr = eps_series[i-1], eps_series[i]
        deltas.append(curr - prev)
        if prev != 0:
            pct_changes.append((curr - prev) / abs(prev))

    avg_eps_magnitude = np.mean([abs(x) for x in eps_series if x is not None])

    if avg_eps_magnitude > threshold and pct_changes:
        # Use average % change
        return np.mean(pct_changes)
    else:
        # Use average delta scaled to forward EPS
        avg_delta = np.mean(deltas)
        return avg_delta / abs(eps_end) if eps_end != 0 else None


# calculate technical and pulling all the fundamental
def analyze_ticker(ticker, etf=False):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo")

    # If no price history, return minimal fields
    if hist.empty:
        base = {
            "Ticker": ticker,
            "Price": "N/A",
            "RSI": "N/A",
            "Momentum": "N/A",
            "Action": "N/A",
        }
        if etf:
            return base
        else:
            return {
                **base,
                "FwdEPS": "N/A",
                "FwdPE": "N/A",
                "P/FCF": "N/A",
                "Growth": "N/A",
                "CAGR": "N/A",
                "MarketCap": "N/A",
                "Sector": "N/A",
                "Beta": "N/A",
                "Target": "N/A",
                # "Method": "N/A",
                # "ShortRatio": "N/A",
            }

    # --- Technicals (always needed for both stock + ETF) ---
    current_price = hist['Close'].iloc[-1]
    rsi = rsi_wilder(hist['Close']).iloc[-1]

    # --- MACD Calculation ---
    try:
        macd, signal_line, hist_line = compute_macd(hist['Close'])
        macd_today = macd.iloc[-1]
        signal_today = signal_line.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_prev = signal_line.iloc[-2]

        momentum = "Bullish" if macd_today >= signal_today else "Bearish"

        if macd_today >= signal_today and macd_prev < signal_prev:
            action = "Buy"
        elif macd_today <= signal_today and macd_prev > signal_prev:
            action = "Sell"
        else:
            action = "No"
    except Exception:
        momentum = "N/A"
        action = "N/A"

    # --- If ETF, return early (no fundamentals) ---
    if etf:
        return {
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "RSI": round(rsi, 2),
            "Momentum": momentum,
            "Action": action,
        }

    # --- Fundamentals (stocks only) ---
    yq = Ticker(ticker)

    eps_next_year, next_year_growth, next_year_revenue, eps_curr_year = None, None, None, None
    try:
        eps_trend = yq.earnings_trend[ticker]['trend']

        curr_year_entry = [e for e in eps_trend if e['period'] == '0y'][0]
        eps_curr_year = curr_year_entry['earningsEstimate']['avg']
        if isinstance(eps_curr_year, dict):
            eps_curr_year = eps_curr_year.get('raw')

        next_year_entry = [e for e in eps_trend if e['period'] == '+1y'][0]
        eps_next_year = next_year_entry['earningsEstimate']['avg']
        if isinstance(eps_next_year, dict):
            eps_next_year = eps_next_year.get('raw')

        next_year_growth = next_year_entry['growth']
        if isinstance(next_year_growth, dict):
            next_year_growth = next_year_growth.get('raw')

        next_year_revenue = next_year_entry.get('revenueEstimate', {}).get('avg')
        if isinstance(next_year_revenue, dict):
            next_year_revenue = next_year_revenue.get('raw')
    except Exception:
        pass

    forward_pe = current_price / eps_next_year if eps_next_year else None

    # Market Cap, Sector, Beta
    try:
        summary = yq.summary_detail[ticker]
        market_cap = summary.get('marketCap', None)

        if market_cap:
            if market_cap < 2e9:
                cap_category = "Small"
            elif market_cap < 10e9:
                cap_category = "Mid"
            else:
                cap_category = "Large"
        else:
            cap_category = "N/A"
    except Exception:
        cap_category = "N/A"

    short_ratio = tk.info.get("shortRatio", None)  # kept for future use
    sector = tk.info.get("sector", None)
    beta = tk.info.get("beta", None)
    shares_outstanding = tk.info.get("sharesOutstanding", None)

    # Manual P/FCF
    try:
        cashflow = tk.cashflow
        fcf_series = cashflow.loc['Free Cash Flow']
        fcf = fcf_series.iloc[0]
        if fcf and shares_outstanding:
            fcf_per_share = fcf / shares_outstanding
            pfcf = current_price / fcf_per_share
        else:
            pfcf = None
    except Exception:
        pfcf = None

    # Valuation logic
    trailing_pe = tk.info.get("trailingPE", None)
    ev_to_rev = tk.info.get("enterpriseToRevenue", None)

    if eps_next_year and eps_next_year > 0 and trailing_pe:
        valuation = eps_next_year * trailing_pe
        valuation_method = "Multiple"
    else:
        valuation = compute_evrev_valuation(
            ev_to_rev, next_year_revenue, shares_outstanding, sector
        )
        valuation_method = "Revenue" if valuation else "N/A"

    # CAGR
    try:
        eps_history = tk.income_stmt.loc['Basic EPS']
        if eps_history.max() > 50:  # ADR mismatch heuristic
            hybrid_cagr = None
        else:
            hybrid_cagr = compute_hybrid_growth(
                eps_history, eps_curr_year, eps_next_year
            )
    except Exception:
        hybrid_cagr = None

    return {
        "Ticker": ticker,
        "Price": round(current_price, 2),
        "RSI": round(rsi, 2),
        "Momentum": momentum,
        "Action": action,
        "FwdPE": round(forward_pe, 2) if forward_pe else "N/A",
        "FwdEPS": round(eps_next_year, 2) if eps_next_year else "N/A",
        "P/FCF": round(pfcf, 2) if pfcf else "N/A",
        "Growth": f"{next_year_growth*100:.2f}%" if next_year_growth else "N/A",
        "CAGR": f"{hybrid_cagr*100:.2f}%" if hybrid_cagr else "N/A",
        "MarketCap": cap_category,
        "Sector": sector or "N/A",
        "Beta": round(beta, 3) if beta else "N/A",
        "Target": round(valuation, 2) if valuation else "N/A",
        # "Method": valuation_method,
        # "ShortRatio": round(short_ratio, 2) if short_ratio else "N/A",
    }


def sort_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort tickers intelligently:
    - If only RSI exists → sort by RSI only (ETF mode)
    - If fundamentals exist → sort by RSI, FwdPE, P/FCF, Growth
    """

    # --- Case 1: ETF mode (only RSI present) ---
    if set(["FwdPE", "P/FCF", "Growth"]).issubset(df.columns) is False:
        return df.sort_values(by="RSI", ascending=True).reset_index(drop=True)

    # --- Case 2: Stock mode (full fundamentals) ---

    # Convert Growth ("10.5%") → float
    df["Growth_num"] = (
        df["Growth"]
        .str.rstrip("%")
        .replace("N/A", None)
    )
    df["Growth_num"] = pd.to_numeric(df["Growth_num"], errors="coerce")

    # Convert P/FCF
    df["P/FCF_num"] = pd.to_numeric(df["P/FCF"], errors="coerce")

    # Sort by RSI, FwdPE, P/FCF, Growth
    df_sorted = df.sort_values(
        by=["RSI", "FwdPE", "P/FCF_num", "Growth_num"],
        ascending=[True, True, True, True]
    )

    # Cleanup
    df_sorted = df_sorted.drop(columns=["Growth_num", "P/FCF_num"])

    return df_sorted.reset_index(drop=True)

def remove_duplicates(tickers):
    """
    Detect and remove duplicate tickers from a list.
    Returns a tuple: (cleaned_list, duplicates_set)
    """
    # Find duplicates
    duplicates = {t for t in tickers if tickers.count(t) > 1}
    
    # Remove duplicates while preserving order
    seen = set()
    cleaned = []
    for t in tickers:
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    
    return cleaned, duplicates

