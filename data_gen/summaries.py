# summaries.py
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Tuple

def load_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_recurring"] = df["is_recurring"].astype(bool)
    return df.dropna(subset=["ts", "amount"])

def _top_k_spend_by_category(df: pd.DataFrame, k: int = 3) -> List[Tuple[str, float]]:
    spend = (
        df[df["amount"] < 0]
        .groupby("category")["amount"].sum().abs()
        .sort_values(ascending=False).head(k)
    )
    return [(cat, round(val, 2)) for cat, val in spend.items()]

def _nz(x: float) -> float:
    # avoid showing -0.0
    return 0.0 if abs(x) < 1e-9 else float(x)

def summarize_window(df: pd.DataFrame, user_id: str, as_of_iso: str, *, days: int) -> Dict:
    """Rolling window summary for the last `days` days ending at as_of (inclusive)."""
    as_of = pd.to_datetime(as_of_iso, utc=True)
    start = as_of - timedelta(days=days)
    u = df[(df["user_id"] == user_id) & (df["ts"] >= start) & (df["ts"] <= as_of)].copy()

    tx_count = int(len(u))
    if tx_count == 0:
        return {
            "user_id": user_id, "as_of": as_of_iso, "window_days": days,
            "income": 0.0, "spend": 0.0, "net": 0.0,
            "top_spend": [], "recurring_total": 0.0, "subs_estimate": 0,
            "delivery_count": 0, "currency": None, "tx_count": 0, "has_data": False,
        }

    currency = u["currency"].mode().iloc[0]

    income = _nz(round(u.loc[u["amount"] > 0, "amount"].sum(), 2))
    spend  = _nz(round(u.loc[u["amount"] < 0, "amount"].sum() * -1, 2))
    net    = _nz(round(income - spend, 2))

    top_spend = _top_k_spend_by_category(u, 3)
    recurring_total = _nz(round(u[(u["is_recurring"]) & (u["amount"] < 0)]["amount"].sum() * -1, 2))

    recurring_df = u[(u["is_recurring"]) & (u["amount"] < 0)]
    if not recurring_df.empty:
        subs_estimate = (
            recurring_df.groupby("merchant")["amount"]
            .apply(lambda s: s.abs().median())
            .pipe(lambda s: (s.between(5, 200)).sum())
        )
    else:
        subs_estimate = 0

    delivery_mask = (u["category"].str.lower() == "dining") & (
        u["merchant"].str.contains("deliver|express", case=False, na=False)
    )
    delivery_count = int(delivery_mask.sum())

    return {
        "user_id": user_id, "as_of": as_of_iso, "currency": currency, "window_days": days,
        "income": income, "spend": spend, "net": net,
        "top_spend": top_spend, "recurring_total": recurring_total,
        "subs_estimate": int(subs_estimate), "delivery_count": delivery_count,
        "tx_count": tx_count, "has_data": True,
    }

def summarize_period(df: pd.DataFrame, user_id: str, as_of_iso: str, period: str) -> Dict:
    """period ∈ {'week','month','year'} → 7/30/365 rolling days."""
    period = period.lower()
    if period == "week":
        return summarize_window(df, user_id, as_of_iso, days=7)
    if period == "month":
        return summarize_window(df, user_id, as_of_iso, days=30)
    if period == "year":
        return summarize_window(df, user_id, as_of_iso, days=365)
    raise ValueError("period must be one of: week, month, year")