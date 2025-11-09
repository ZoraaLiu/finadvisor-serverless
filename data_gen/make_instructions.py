# make_instructions.py
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
import yaml

from summaries import summarize_window, summarize_period
from baseline import baseline_tips


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

    _req(cfg, "random_seed", int)
    outs = _req(cfg, "outputs", dict)
    _req(outs, "train", str)
    _req(outs, "eval", str)

    split = _req(cfg, "split", dict)
    _req(split, "train_frac", float)

    # optional limit (allow null)
    if "users_limit" not in cfg or cfg["users_limit"] is None:
        cfg["users_limit"] = None
    elif not isinstance(cfg["users_limit"], int):
        raise ValueError("users_limit must be int or null")

    as_of = _req(cfg, "as_of", dict)
    _req(as_of, "count", int)
    _req(as_of, "step_days", int)

    _req(cfg, "periods", list)
    _req(cfg, "custom_days", list)

    return cfg


# ------------------- text builders -------------------

SYSTEM = "You are a budgeting assistant. Be practical, safe, and concise."

def _fmt(cur: str, x: float) -> str:
    x = 0.0 if abs(x) < 1e-9 else x
    s = f"{x:,.0f}"
    return f"{cur} {s}" if cur else s

def _valid_summary(s: dict) -> bool:
    if not s.get("has_data"):
        return False
    if s.get("tx_count", 0) < 2:
        return False
    if (s.get("income", 0.0) == 0.0) and (s.get("spend", 0.0) == 0.0):
        return False
    return True

def _persona_from_summary(s: dict) -> str:
    days = int(s.get("window_days", 30) or 30)
    spend = float(s.get("spend", 0.0))
    income = float(s.get("income", 0.0))
    top = {k.lower(): v for k, v in (s.get("top_spend") or [])}
    dining = top.get("dining", 0.0)
    transport = top.get("transport", 0.0)
    rent = top.get("rent", 0.0)
    recurring = float(s.get("recurring_total", 0.0))
    subs = int(s.get("subs_estimate", 0))
    delivery = int(s.get("delivery_count", 0))

    def monthly(x): 
        return x * (30.0 / max(1, days))

    spend_m = monthly(spend) if spend > 0 else 0
    dining_share = (dining / spend) if spend > 0 else 0
    transport_share = (transport / spend) if spend > 0 else 0
    rent_share = (rent / spend) if spend > 0 else 0
    recurring_share = (recurring / max(1e-6, income)) if income > 0 else 0

    if rent_share > 0.35 or rent > 12000:
        base = "rent-heavy professional"
    elif dining_share > 0.25 or delivery >= 4:
        base = "foodie with frequent deliveries"
    elif subs >= 3 or recurring_share > 0.10:
        base = "subscription-maximizer"
    elif transport_share > 0.20:
        base = "daily commuter"
    elif spend_m < 6000 and income > 0:
        base = "frugal saver"
    else:
        base = random.choice([
            "urban worker", "busy professional", "city dweller",
            "budget-conscious user", "everyday spender"
        ])
    return base

TASK_TEMPLATES = [
    "Task: Give 2-3 concrete budgeting tips with rough {cur} savings per month.",
    "Task: Provide 2-3 practical budgeting suggestions with estimated {cur} monthly savings.",
    "Task: Suggest 2-3 actionable steps to reduce spend with ~{cur} monthly savings estimates.",
]

def build_input_text(summary: Dict) -> str:
    cur = summary.get("currency") or ""
    days = int(summary.get("window_days", 90))
    top = ", ".join([f"{k}:{v:.0f}" for k, v in (summary.get("top_spend") or [])]) or "—"

    inc = float(summary.get("income", 0.0))
    spd = float(summary.get("spend", 0.0))
    net = float(summary.get("net", inc - spd))
    rec = float(summary.get("recurring_total", 0.0))
    subs = int(summary.get("subs_estimate", 0))
    delivery = int(summary.get("delivery_count", 0))

    profile = _persona_from_summary(summary)
    task_line = random.choice(TASK_TEMPLATES).format(cur=(cur + " ") if cur else "")

    text = (
        f"Profile: {profile}.\n"
        f"Window: last {days} days.\n"
        f"Totals: Income {_fmt(cur, inc)}; Spend {_fmt(cur, spd)}; Net {_fmt(cur, net)}.\n"
        f"Top spend: {top}.\n"
        f"Recurring: {_fmt(cur, rec)} across {subs} items.\n"
        f"Delivery orders (approx): {delivery}.\n"
        f"{task_line}"
    )
    return text.replace("  ", " ").strip()


# ------------------- generation core -------------------

def _iter_as_of_dates(df: pd.DataFrame, count: int, step_days: int) -> List[str]:
    ts_max = pd.to_datetime(df["ts"], utc=True).max()
    if pd.isna(ts_max):
        return []
    dates = []
    for i in range(count):
        d = ts_max - pd.Timedelta(days=i * step_days)
        dates.append(d.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return dates

def _generate_rows(
    df: pd.DataFrame,
    users: Iterable[str],
    as_of_list: Iterable[str],
    periods: Iterable[str],
    custom_days: Iterable[int],
) -> List[Dict]:
    rows: List[Dict] = []
    seen: set = set()
    for u in users:
        for as_of in as_of_list:
            # Period-based summaries
            for p in periods:
                try:
                    s = summarize_period(df, u, as_of, p)
                except Exception:
                    continue
                if not _valid_summary(s):
                    continue
                key = (u, as_of, int(s.get("window_days", 0)))
                if key in seen:
                    continue
                seen.add(key)
                inp = build_input_text(s)
                tips = baseline_tips(s)
                out = "- " + "\n- ".join(tips)
                rows.append({
                    "user_id": u,
                    "as_of": as_of,
                    "window_days": s.get("window_days"),
                    "input": inp,
                    "output": out
                })

            # Custom-day windows
            for d in custom_days:
                try:
                    s = summarize_window(df, u, as_of, days=int(d))
                except Exception:
                    continue
                if not _valid_summary(s):
                    continue
                key = (u, as_of, int(s.get("window_days", 0)))
                if key in seen:
                    continue
                seen.add(key)
                inp = build_input_text(s)
                tips = baseline_tips(s)
                out = "- " + "\n- ".join(tips)
                rows.append({
                    "user_id": u,
                    "as_of": as_of,
                    "window_days": s.get("window_days"),
                    "input": inp,
                    "output": out
                })
    return rows

def build_from_yaml(config_path: str, csv_path: str) -> tuple[str, str]:
    """
    Build instructions/eval from YAML + transactions CSV.
    Returns (train_path, eval_path).
    """
    cfg = load_config(Path(config_path))

    # seed for reproducibility
    random.seed(int(cfg["random_seed"]))

    # load + normalize transactions
    df = pd.read_csv(csv_path)
    if "ts" not in df.columns or "user_id" not in df.columns:
        raise ValueError("CSV must include 'ts' and 'user_id' columns")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    users = sorted(df["user_id"].unique())
    if cfg["users_limit"]:
        users = users[: int(cfg["users_limit"])]
    if not users:
        raise RuntimeError("No users found in CSV after loading.")

    as_of_cfg = cfg["as_of"]
    as_of_list = _iter_as_of_dates(df, as_of_cfg["count"], as_of_cfg["step_days"])
    if not as_of_list:
        raise RuntimeError("No as_of dates computed (check 'ts' parsing).")

    periods = list(cfg["periods"])
    custom_days = list(cfg["custom_days"])

    rows = _generate_rows(df, users, as_of_list, periods, custom_days)
    if not rows:
        raise RuntimeError("No rows generated. Check your data and YAML settings.")

    random.shuffle(rows)
    split = int(len(rows) * float(cfg["split"]["train_frac"]))
    train_rows = rows[:split]
    eval_rows = rows[split:]

    out_train = Path(cfg["outputs"]["train"])
    out_eval = Path(cfg["outputs"]["eval"])
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_eval.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({"input": r["input"], "output": r["output"]}, ensure_ascii=False) + "\n")

    with open(out_eval, "w", encoding="utf-8") as f:
        for i, r in enumerate(eval_rows, 1):
            f.write(json.dumps({"prompt_id": f"p{i:03d}", "input": r["input"]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} train pairs → {out_train}")
    print(f"Wrote {len(eval_rows)} eval prompts → {out_eval}")
    print(f"(Users: {len(users)}, As-of: {len(as_of_list)}, Periods: {periods}, Custom days: {custom_days})")

    return str(out_train), str(out_eval)