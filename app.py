import math
import random
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import requests
import pandas as pd
import streamlit as st


# =========================
# API endpoints
# =========================
# Forbidden jewel dataset endpoint (the one you found via DevTools)
FORBIDDEN_URL = "https://poe.ninja/poe1/api/economy/stash/current/item/overview"

# Currency endpoint for Divine->Chaos conversion [1](https://www.reddit.com/r/pathofexile/comments/sp5jd0/forbidden_flesh_forbidden_flame_how_do_you_search/)
CURRENCY_URL = "https://poe.ninja/api/data/currencyoverview"


# =========================
# Data models
# =========================
@dataclass
class Outcome:
    passive: str
    price_chaos: float
    listings: float
    ascendancy: Optional[str] = None
    base_class: Optional[str] = None


# =========================
# Helpers
# =========================
def weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    denom = sum(weights)
    if denom <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / denom


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile. p in [0,1]."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    idx = p * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": med,
        "min": s[0],
        "max": s[-1],
        "p10": percentile(s, 0.10),
        "p25": percentile(s, 0.25),
        "p50": percentile(s, 0.50),
        "p75": percentile(s, 0.75),
        "p90": percentile(s, 0.90),
    }


# =========================
# Fetch functions (cached)
# =========================
@st.cache_data(ttl=300, show_spinner=False)  # cache 5 minutes to be nice to poe.ninja
def fetch_forbidden_lines(league: str) -> List[dict]:
    r = requests.get(FORBIDDEN_URL, params={"league": league, "type": "ForbiddenJewel"}, timeout=30)
    r.raise_for_status()
    return r.json().get("lines", [])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_divine_chaos_equivalent(league: str) -> float:
    """
    Reads Divine Orb chaosEquivalent from poe.ninja currencyoverview. [1](https://www.reddit.com/r/pathofexile/comments/sp5jd0/forbidden_flesh_forbidden_flame_how_do_you_search/)
    """
    r = requests.get(CURRENCY_URL, params={"league": league, "type": "Currency"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    for line in data.get("lines", []):
        if line.get("currencyTypeName") == "Divine Orb":
            return float(line["chaosEquivalent"])
    raise RuntimeError("Divine Orb not found in currencyoverview response")


# =========================
# Build distribution + filters
# =========================
def build_distribution(
    lines: List[dict],
    variant_label: str,
    min_listings: float,
    exclude_names: set
) -> List[Outcome]:
    dist: List[Outcome] = []
    for line in lines:
        if line.get("variant") != variant_label:
            continue

        price = line.get("chaosValue")
        if price is None:
            continue

        listings = float(line.get("listingCount", 0) or 0)
        if listings < min_listings:
            continue

        md = line.get("metadata", {}) or {}
        passive = md.get("passiveName") or line.get("name")
        if not passive:
            continue

        if passive in exclude_names:
            continue

        dist.append(Outcome(
            passive=passive,
            price_chaos=float(price),
            listings=listings,
            ascendancy=md.get("ascendancy"),
            base_class=md.get("baseClass"),
        ))
    return dist


def apply_trims(dist: List[Outcome], trim_low: float, trim_high: float, drop_top_n: int) -> List[Outcome]:
    if not dist:
        return dist

    trim_low = max(0.0, min(trim_low, 0.49))
    trim_high = max(0.0, min(trim_high, 0.49))

    dist_sorted = sorted(dist, key=lambda x: x.price_chaos)
    n = len(dist_sorted)

    lo = int(round(n * trim_low))
    hi = int(round(n * (1 - trim_high)))
    hi = max(hi, lo)

    trimmed = dist_sorted[lo:hi]

    if drop_top_n > 0 and len(trimmed) > drop_top_n:
        trimmed = trimmed[:-drop_top_n]

    return trimmed


# =========================
# EV + profitability
# =========================
def compute_ev(dist: List[Outcome], model: str) -> Optional[float]:
    prices = [x.price_chaos for x in dist]
    if not prices:
        return None
    if model == "uniform":
        return sum(prices) / len(prices)
    if model == "listings":
        weights = [x.listings for x in dist]
        return weighted_mean(prices, weights)
    raise ValueError("model must be uniform or listings")


def chance_profit_single(dist: List[Outcome], buy_price_chaos: float, model: str) -> Optional[float]:
    if not dist:
        return None
    if model == "uniform":
        return sum(1 for x in dist if x.price_chaos > buy_price_chaos) / len(dist)
    if model == "listings":
        win_w = sum(x.listings for x in dist if x.price_chaos > buy_price_chaos)
        total_w = sum(x.listings for x in dist)
        return (win_w / total_w) if total_w > 0 else None
    return None


def simulate_runs(
    dist: List[Outcome],
    buy_price_chaos: float,
    model: str,
    n_buys: int,
    trials: int,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Returns (total_values, profits) for each trial.
    profit = sum(revealed values) - n_buys * buy_price
    """
    if seed is not None:
        random.seed(seed)

    prices = [x.price_chaos for x in dist]
    weights = [x.listings for x in dist]

    totals = []
    profits = []

    total_cost = n_buys * buy_price_chaos

    if model == "uniform":
        for _ in range(trials):
            total_value = sum(random.choice(prices) for _ in range(n_buys))
            totals.append(total_value)
            profits.append(total_value - total_cost)
    else:  # listings
        for _ in range(trials):
            drawn = random.choices(prices, weights=weights, k=n_buys)
            total_value = sum(drawn)
            totals.append(total_value)
            profits.append(total_value - total_cost)

    return totals, profits


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Forbidden Jewel EV + Profitability", layout="wide")
st.title("Forbidden Jewel EV + Profitability (poe.ninja)")
st.caption("Uses poe.ninja ForbiddenJewel prices and simulates buying unidentified jewels (Monte Carlo).")

with st.sidebar:
    st.header("Inputs")

    league = st.text_input("League", value="Mirage")

    which = st.selectbox("Jewel Type", ["Forbidden Flesh", "Forbidden Flame"])

    price_mode = st.radio("Buy price input", ["Chaos", "Divines"], horizontal=True)

    if price_mode == "Chaos":
        buy_price_input = st.number_input("Unidentified buy price (chaos)", min_value=0.0, value=60.0, step=1.0)
        buy_price_chaos = buy_price_input
        divine_rate = None
    else:
        buy_price_input = st.number_input("Unidentified buy price (divines)", min_value=0.0, value=9.0, step=0.5)
        divine_rate = fetch_divine_chaos_equivalent(league)
        buy_price_chaos = buy_price_input * divine_rate

    model = st.selectbox("Probability model", ["listings", "uniform"], index=0,
                         help="listings = weighted by listingCount (market-weighted); uniform = each outcome equally likely.")

    st.subheader("Outlier / filter controls")
    min_listings = st.number_input("Min listingCount", min_value=0.0, value=0.0, step=1.0,
                                   help="Ignore outcomes with listingCount less than this.")

    trim_low = st.slider("Trim lowest % by price", 0.0, 20.0, 0.0, step=0.5) / 100.0
    trim_high = st.slider("Trim highest % by price", 0.0, 20.0, 0.0, step=0.5) / 100.0

    drop_top_n = st.number_input("Drop top N most expensive outcomes", min_value=0, value=0, step=1)

    exclude_text = st.text_area(
        "Exclude passives (one per line)",
        value="",
        height=130,
        help="Paste passive names exactly (e.g. Indomitable Resolve)."
    )

    st.subheader("Simulation")
    n_buys = st.slider("Number of unid jewels per run", 1, 200, 20)
    trials = st.slider("Number of simulation trials", 1000, 50000, 20000, step=1000)
    seed = st.text_input("Seed (optional, for reproducible runs)", value="")

    run_btn = st.button("Run / Refresh", type="primary")


# Run automatically on first load too
if run_btn or True:
    try:
        lines = fetch_forbidden_lines(league)

        exclude_names = {ln.strip() for ln in exclude_text.splitlines() if ln.strip()}

        dist = build_distribution(
            lines=lines,
            variant_label=which,
            min_listings=min_listings,
            exclude_names=exclude_names
        )

        dist = apply_trims(dist, trim_low, trim_high, int(drop_top_n))

        if not dist:
            st.error("No outcomes left after filtering. Reduce excludes/trims or lower min listingCount.")
            st.stop()

        ev = compute_ev(dist, model)
        p_single = chance_profit_single(dist, buy_price_chaos, model)

        prices = [x.price_chaos for x in dist]
        stats = summarize(prices)

        # Simulation
        seed_val = int(seed) if seed.strip().isdigit() else None
        totals, profits = simulate_runs(dist, buy_price_chaos, model, n_buys, trials, seed_val)
        profits_sorted = sorted(profits)

        prob_profitable = sum(1 for p in profits if p > 0) / len(profits)
        avg_profit = sum(profits) / len(profits)
        med_profit = profits_sorted[len(profits_sorted) // 2] if len(profits_sorted) % 2 else (
            profits_sorted[len(profits_sorted)//2 - 1] + profits_sorted[len(profits_sorted)//2]
        ) / 2

        p10_profit = percentile(profits_sorted, 0.10)
        p25_profit = percentile(profits_sorted, 0.25)
        p50_profit = percentile(profits_sorted, 0.50)
        p75_profit = percentile(profits_sorted, 0.75)
        p90_profit = percentile(profits_sorted, 0.90)

        total_cost = n_buys * buy_price_chaos
        avg_value = sum(totals) / len(totals)
        avg_roi = (avg_value / total_cost - 1) * 100 if total_cost > 0 else None

        # =========================
        # Display
        # =========================
        top = st.columns(4)
        top[0].metric("Buy price (chaos)", f"{buy_price_chaos:,.1f}c")
        if divine_rate is not None:
            top[1].metric("Divine rate (c/div)", f"{divine_rate:,.1f}c")  # from poe.ninja currencyoverview [1](https://www.reddit.com/r/pathofexile/comments/sp5jd0/forbidden_flesh_forbidden_flame_how_do_you_search/)
        else:
            top[1].metric("Divine rate (c/div)", "—")

        top[2].metric("EV per revealed jewel", f"{ev:,.2f}c")
        top[3].metric("EV profit per jewel", f"{(ev - buy_price_chaos):,.2f}c")

        st.divider()

        mid = st.columns(4)
        mid[0].metric("Single-roll chance profit", f"{(p_single or 0)*100:.2f}%")
        mid[1].metric(f"Chance profitable after {n_buys} buys", f"{prob_profitable*100:.2f}%")
        mid[2].metric("Average profit (per run)", f"{avg_profit:,.1f}c")
        mid[3].metric("Median profit (per run)", f"{med_profit:,.1f}c")

        st.markdown("### Outcome price distribution (per revealed jewel)")
        dist_cols = st.columns(5)
        dist_cols[0].metric("Median", f"{stats['median']:,.1f}c")
        dist_cols[1].metric("P10", f"{stats['p10']:,.1f}c")
        dist_cols[2].metric("P90", f"{stats['p90']:,.1f}c")
        dist_cols[3].metric("Min", f"{stats['min']:,.1f}c")
        dist_cols[4].metric("Max", f"{stats['max']:,.1f}c")

        st.markdown("### Profit distribution (simulation)")
        prof_cols = st.columns(5)
        prof_cols[0].metric("P10 profit", f"{p10_profit:,.1f}c")
        prof_cols[1].metric("P25 profit", f"{p25_profit:,.1f}c")
        prof_cols[2].metric("P50 profit", f"{p50_profit:,.1f}c")
        prof_cols[3].metric("P75 profit", f"{p75_profit:,.1f}c")
        prof_cols[4].metric("P90 profit", f"{p90_profit:,.1f}c")

        st.caption(f"Average ROI per run: {avg_roi:.2f}% (based on mean simulated value / total cost)")

        # Histogram chart using pandas
        df_profit = pd.DataFrame({"profit_chaos": profits})
        st.bar_chart(df_profit["profit_chaos"].value_counts(bins=60).sort_index())

        st.divider()

        # Show cheapest / most expensive outcomes used
        dist_sorted = sorted(dist, key=lambda x: x.price_chaos)
        show_n = 10

        left, right = st.columns(2)

        with left:
            st.subheader(f"Cheapest {show_n} outcomes (after filters)")
            df_low = pd.DataFrame([{
                "passive": x.passive,
                "price_chaos": x.price_chaos,
                "listings": x.listings,
                "ascendancy": x.ascendancy,
                "baseClass": x.base_class
            } for x in dist_sorted[:show_n]])
            st.dataframe(df_low, use_container_width=True)

        with right:
            st.subheader(f"Most expensive {show_n} outcomes (after filters)")
            df_high = pd.DataFrame([{
                "passive": x.passive,
                "price_chaos": x.price_chaos,
                "listings": x.listings,
                "ascendancy": x.ascendancy,
                "baseClass": x.base_class
            } for x in dist_sorted[-show_n:][::-1]])
            st.dataframe(df_high, use_container_width=True)

        # Download CSV of outcomes used
        st.markdown("### Download")
        df_all = pd.DataFrame([{
            "passive": x.passive,
            "variant": which,
            "price_chaos": x.price_chaos,
            "listings": x.listings,
            "ascendancy": x.ascendancy,
            "baseClass": x.base_class
        } for x in dist_sorted])

        csv_buf = io.StringIO()
        df_all.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download outcomes CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"{league}_{which.replace(' ', '_')}_outcomes.csv",
            mime="text/csv"
        )

    except requests.HTTPError as e:
        st.error(f"HTTP error from poe.ninja: {e}")
    except Exception as e:
        st.error(f"Error: {e}")