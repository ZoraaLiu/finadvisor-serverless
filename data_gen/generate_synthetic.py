import os
import json
import random
import datetime
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, "profiles.yaml"), "r") as f:
    config = yaml.safe_load(f)

CATEGORIES = config["categories"]


def random_date_in_last_n_days(n=45):
    today = datetime.date.today()
    delta = datetime.timedelta(days=random.randint(0, n))
    d = today - delta
    return d.isoformat()


def generate_txns_for_user(user_cfg, n_txns=350):
    income = user_cfg["monthly_income"]
    txns = []
    # add income event
    txns.append({
        "date": random_date_in_last_n_days(30),
        "type": "income",
        "amount": income,
        "description": "Monthly salary",
        "category": "income"
    })
    for _ in range(n_txns):
        cat = random.choice(CATEGORIES)
        # very rough amounts by category
        base = {
            "food": (8, 45),
            "rent": (800, 1500),
            "shopping": (20, 150),
            "transport": (5, 50),
            "subscriptions": (5, 30),
            "health": (30, 120),
        }.get(cat, (10, 100))
        amount = round(random.uniform(*base), 2)
        txns.append({
            "date": random_date_in_last_n_days(),
            "type": "expense",
            "amount": amount,
            "description": f"{cat} purchase",
            "category": cat
        })
    return txns


def main():
    for user in config["users"]:
        txns = generate_txns_for_user(user)
        out_path = os.path.join(OUT_DIR, f"txns_{user['id']}.json")
        with open(out_path, "w") as f:
            json.dump(txns, f, indent=2)
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()