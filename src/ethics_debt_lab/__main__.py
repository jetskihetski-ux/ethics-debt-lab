from __future__ import annotations

import argparse

from .simulation import run_simulation


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic insurance ethics debt simulation."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--customers", type=int, default=5000)
    args = parser.parse_args()

    results = run_simulation(args.customers, args.seed)
    print(
        f"{'strategy':<18} {'profit delta':>14} {'harm score':>12} "
        f"{'lapse rate':>12} {'verdict':>10}"
    )
    print("-" * 72)
    for result in results:
        print(
            f"{result['strategy']:<18} "
            f"{money(float(result['profit_delta'])):>14} "
            f"{float(result['consumer_harm']):>12.1f} "
            f"{float(result['lapse_rate']):>11.1%} "
            f"{result['verdict']:>10}"
        )


if __name__ == "__main__":
    main()
