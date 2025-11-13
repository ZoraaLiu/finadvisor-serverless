"""
Pipeline for data generation and instruction building (in-process).
Steps:
  1) generate_transactions.generate_from_yaml(config)  → CSV
  2) make_instructions.build_from_yaml(config, CSV)    → JSONL files
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

from generate_transactions import generate_from_yaml
from make_instructions  import build_from_yaml

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def step_generate(gen_cfg: Path, out_csv: Path) -> Path:
    """Generate synthetic transactions from YAML config."""
    ensure_parent(out_csv)
    csv_path = generate_from_yaml(str(gen_cfg), out_csv=str(out_csv))
    print(f"Generated CSV → {csv_path}")
    return Path(csv_path)

def step_prompts(instr_cfg: Path, csv_path: Path) -> tuple[Path, Path]:
    """Build instructions/eval sets from YAML config + transactions CSV."""
    if not csv_path.exists():
        sys.exit(f"Missing transactions file: {csv_path} (run --generate first or pass --csv)")
    train_path, eval_path = build_from_yaml(str(instr_cfg), str(csv_path))
    print(f"Built instructions → {train_path}")
    print(f"Built eval prompts → {eval_path}")
    return Path(train_path), Path(eval_path)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data generation + prompt building pipeline (YAML-driven)")

    p.add_argument("--all",       action="store_true", help="Run both steps")
    p.add_argument("--generate",  action="store_true", help="Run data generation only")
    p.add_argument("--prompts",   action="store_true", help="Run prompt building only")

    # Config paths (YAML)
    p.add_argument("--gen-config",   type=Path, default=Path("configs/generate.yaml"),
                   help="YAML config for transaction generation")
    p.add_argument("--instr-config", type=Path, default=Path("configs/instructions.yaml"),
                   help="YAML config for instructions/eval generation")

    # I/O paths
    p.add_argument("--out-csv", type=Path, default=Path("data/transactions.csv"),
                   help="Where to write generated transactions CSV")
    p.add_argument("--csv",     type=Path, default=None,
                   help="Use an existing CSV instead of generating (overrides --out-csv for prompts step)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        args.generate = args.prompts = True

    if not (args.generate or args.prompts):
        print("Nothing to do. Use --all or pick steps with --generate / --prompts.")
        return

    csv_for_prompts: Path | None = None
    if args.generate:
        if not args.gen_config.exists():
            sys.exit(f"Missing generation config: {args.gen_config}")
        csv_for_prompts = step_generate(args.gen_config, args.out_csv)

    if args.prompts:
        if not args.instr_config.exists():
            sys.exit(f"Missing instructions config: {args.instr_config}")
        csv_path = args.csv or csv_for_prompts or args.out_csv
        step_prompts(args.instr_config, csv_path)


if __name__ == "__main__":
    main()