from typing import Dict, List
import math

def _fmt_currency(x: float, cur: str) -> str:
    return f"{cur} {x:,.0f}" if cur else f"{x:,.0f}"

def _monthlyize(amount: float, window_days: int) -> float:
    if window_days <= 0:
        return amount
    return amount * (30.0 / window_days)

def _cap_saving(save: float, spend_m: float) -> float:
    upper = max(50.0, 0.30 * spend_m)
    return max(20.0, min(save, upper))

def _get_cat_amount(top_spend, names) -> float:
    names_l = {n.lower() for n in names}
    for k, v in top_spend:
        if k.lower() in names_l:
            return float(v)
    return 0.0

def baseline_tips(summary: Dict) -> List[str]:
    cur = summary.get("currency") or ""
    income_win = float(summary.get("income", 0.0))
    spend_win  = float(summary.get("spend", 0.0))
    net_win    = float(summary.get("net", income_win - spend_win))
    window_days = int(summary.get("window_days", 90))

    top_spend = summary.get("top_spend", [])
    recurring_total_win = float(summary.get("recurring_total", 0.0))
    subs_est = int(summary.get("subs_estimate", 0))
    delivery = int(summary.get("delivery_count", 0))

    income_m    = _monthlyize(income_win, window_days)
    spend_m     = _monthlyize(spend_win,  window_days)
    recurring_m = _monthlyize(recurring_total_win, window_days)

    dining_win   = _get_cat_amount(top_spend, {"dining", "food", "food & drink"})
    shopping_win = _get_cat_amount(top_spend, {"shopping"})
    rent_win     = _get_cat_amount(top_spend, {"rent"})

    dining_share   = (dining_win / spend_win) if spend_win > 0 else 0.0
    shopping_share = (shopping_win / spend_win) if spend_win > 0 else 0.0
    rent_share     = (rent_win / spend_win) if spend_win > 0 else 0.0

    tips: List[str] = []

    # Basic flags
    overspending = income_m > 0 and spend_m > income_m
    recurring_ratio = (recurring_m / income_m) if income_m > 0 else 0.0

    # 1) Savings rate nudge (target ≈ 10%)
    if income_m > 0 and not overspending:
        current_savings_m = _monthlyize(net_win, window_days)
        current_savings_rate = max(0.0, current_savings_m / income_m)
        if current_savings_rate < 0.10:
            target = _cap_saving(round(income_m * 0.10), spend_m)
            tips.append(
                f"Automate a transfer the day after payday: move ~{_fmt_currency(target, cur)}/month to savings."
            )

    # 2) Recurring/subscriptions check
    if income_m > 0 and recurring_m >= 50:
        if recurring_ratio > 0.08:
            est = _cap_saving(round(recurring_m * 0.10), spend_m)
            tips.append(
                f"Recurring charges look high; cancel one unused subscription to save ~{_fmt_currency(est, cur)}/month."
            )
        elif subs_est >= 3:
            est = _cap_saving(round(recurring_m * 0.06), spend_m)
            tips.append(
                f"You have {subs_est} recurring items; audit and drop one to save ~{_fmt_currency(est, cur)}/month."
            )

    # 3) Dining clamp (cap ~20% of spend)
    if spend_m > 0 and dining_share > 0.20:
        excess = (dining_share - 0.20) * spend_m
        save = _cap_saving(round(excess * 0.5), spend_m)
        tips.append(
            f"Dining is {int(dining_share*100)}% of spend; set a weekly budget to save ~{_fmt_currency(save, cur)}/month."
        )

    # 4) Shopping clamp (cap ~18%)
    if spend_m > 0 and shopping_share > 0.18:
        excess = (shopping_share - 0.18) * spend_m
        save = _cap_saving(round(excess * 0.5), spend_m)
        tips.append(
            f"Shopping is {int(shopping_share*100)}% of spend; 48-hour rule + monthly cap can save ~{_fmt_currency(save, cur)}/month."
        )

    # 5) Rent-heavy situation
    if spend_m > 0 and rent_share > 0.40:
        tips.append(
            "Rent is taking a large share of your budget; consider options like a roommate, "
            "renegotiating your lease, or planning for a move when feasible."
        )

    # 6) Delivery frequency
    weeks = max(1, math.ceil(window_days / 7))
    delivery_per_week = delivery / weeks
    if delivery_per_week > 1.5:  # more than ~1–2 deliveries per week on average
        tips.append(
            "Limit food delivery to once per week and batch-cook extra portions to cut delivery fees and service charges."
        )

    # 7) Overspending fallback
    if not tips and overspending:
        gap = spend_m - income_m
        tips.append(
            f"You're spending about {_fmt_currency(gap, cur)} more than you earn each month; "
            "pick one or two big categories to reduce by 10–15%."
        )

    if not tips:
        tips = [
            "Review your two biggest categories and set weekly envelopes.",
            "Automate a small monthly transfer to savings (start with 5%).",
        ]
    return tips[:3]