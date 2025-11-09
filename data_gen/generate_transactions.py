import argparse
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml

def _req(d: dict, key: str, typ=None):
    if key not in d:
        raise ValueError(f"Missing required config key: '{key}'")
    val = d[key]
    if typ and not isinstance(val, typ):
        raise ValueError(f"Config key '{key}' must be {typ.__name__}")
    return val

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # top-level
    _req(cfg, "random_seed", int)
    _req(cfg, "np_seed", int)
    _req(cfg, "categories", list)
    _req(cfg, "merchants", dict)
    _req(cfg, "users", dict)
    _req(cfg, "transactions", dict)

    # users
    users = cfg["users"]
    _req(users, "count", int)
    _req(users, "currencies", list)
    _req(users, "income_range", list)

    # transactions
    tx = cfg["transactions"]
    _req(tx, "months", int)
    _req(tx, "rent_range", list)
    _req(tx, "subscriptions", dict)
    _req(tx, "variable_spend", dict)

    subs = tx["subscriptions"]
    _req(subs, "count_range", list)
    _req(subs, "amount_range", list)
    _req(subs, "merchants", list)

    vs = tx["variable_spend"]
    _req(vs, "per_month_range", list)
    _req(vs, "category_amounts", dict)

    return cfg


# ------------------- helpers -------------------

def _first_day_n_months_ago(n: int) -> datetime:
    """First day of the month going back n-1 months so we cover exactly n months."""
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    # move back n-1 months
    m = month - (n - 1)
    while m <= 0:
        m += 12
        year -= 1
    return datetime(year, m, 1, tzinfo=timezone.utc)

def _pick_merchant(cfg: dict, cat: str) -> str:
    return random.choice(cfg["merchants"].get(cat, ["General Store"]))

def _make_users(cfg: dict) -> List[Dict]:
    lo, hi = cfg["users"]["income_range"]
    out = []
    for i in range(cfg["users"]["count"]):
        out.append({
            "user_id": f"user_{i+1}",
            "currency": random.choice(cfg["users"]["currencies"]),
            "base_income": int(random.randint(int(lo), int(hi))),
        })
    return out

def _simulate_user(user: Dict, cfg: dict, start_month: datetime, months: int) -> List[Dict]:
    rows: List[Dict] = []
    uid = user["user_id"]
    cur = user["currency"]
    base_income = user["base_income"]

    rent_lo, rent_hi = cfg["transactions"]["rent_range"]
    rent_amt = -int(random.randint(int(rent_lo), int(rent_hi)))

    subs_cfg = cfg["transactions"]["subscriptions"]
    sub_n_lo, sub_n_hi = subs_cfg["count_range"]
    sub_amt_lo, sub_amt_hi = subs_cfg["amount_range"]
    sub_merchants = subs_cfg["merchants"]
    subs = [(-int(random.randint(int(sub_amt_lo), int(sub_amt_hi))), random.choice(sub_merchants))
            for _ in range(random.randint(int(sub_n_lo), int(sub_n_hi)))]

    vs_cfg = cfg["transactions"]["variable_spend"]
    tx_lo, tx_hi = vs_cfg["per_month_range"]
    cat_amounts = vs_cfg["category_amounts"]

    for m in range(months):
        month_start = (pd.Timestamp(start_month) + pd.DateOffset(months=m)).to_pydatetime()

        # income (positive)
        rows.append({
            "user_id": uid,
            "ts": (month_start + timedelta(days=1, hours=9)).astimezone(timezone.utc).isoformat(),
            "amount": float(abs(base_income)), "currency": cur,
            "category": "Income", "merchant": "Employer Ltd", "is_recurring": True,
        })

        # rent (negative)
        rows.append({
            "user_id": uid,
            "ts": (month_start + timedelta(days=2, hours=10)).astimezone(timezone.utc).isoformat(),
            "amount": float(rent_amt), "currency": cur,
            "category": "Rent", "merchant": random.choice(cfg["merchants"]["Rent"]),
            "is_recurring": True,
        })

        # subscriptions (negative)
        for amt, merch in subs:
            rows.append({
                "user_id": uid,
                "ts": (month_start + timedelta(days=random.randint(3, 10), hours=8))
                      .astimezone(timezone.utc).isoformat(),
                "amount": float(amt), "currency": cur,
                "category": "Utilities", "merchant": merch, "is_recurring": True,
            })

        # variable spend (negative)
        for _ in range(random.randint(int(tx_lo), int(tx_hi))):
            cat = random.choice(cfg["categories"])
            if cat == "Rent":  # avoid extra rent in variable
                continue
            lo, hi = cat_amounts.get(cat, [40, 400])
            amt = -float(random.randint(int(lo), int(hi)))
            rows.append({
                "user_id": uid,
                "ts": (month_start + timedelta(days=random.randint(1, 27),
                                               hours=random.randint(8, 21)))
                      .astimezone(timezone.utc).isoformat(),
                "amount": amt, "currency": cur,
                "category": cat, "merchant": _pick_merchant(cfg, cat),
                "is_recurring": False,
            })

    return rows

def generate_from_yaml(config_path: str, out_csv: str) -> str:
    """
    Generate transactions from your YAML config.
    Returns the written CSV path.
    """
    cfg = load_config(Path(config_path))

    # seed exactly as specified
    random.seed(int(cfg["random_seed"]))
    np.random.seed(int(cfg["np_seed"]))

    months = int(cfg["transactions"]["months"])
    start_month = _first_day_n_months_ago(months)

    users = _make_users(cfg)

    all_rows: List[Dict] = []
    for u in users:
        all_rows.extend(_simulate_user(u, cfg, start_month, months))

    df = pd.DataFrame(all_rows, columns=[
        "user_id", "ts", "amount", "currency", "category", "merchant", "is_recurring"
    ])
    # canonicalize timestamp format
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} for {len(users)} users across {months} months.")
    return str(out)