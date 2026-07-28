from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Customer:
    expected_loss: float
    income_band: int
    tenure: float
    shopping_friction: float
    prior_increase: float
    renewed: int


@dataclass(frozen=True)
class Strategy:
    name: str
    base_increase: float
    loyalty_weight: float = 0.0
    friction_weight: float = 0.0
    claim_suppression: float = 0.0
    deductible_shift: float = 0.0


STRATEGIES = (
    Strategy("baseline", 0.03),
    Strategy("loyalty_tax", 0.03, loyalty_weight=0.16),
    Strategy("complexity_rent", 0.03, friction_weight=0.14),
    Strategy("claims_friction", 0.03, claim_suppression=0.18),
    Strategy("coverage_shrink", 0.01, deductible_shift=0.22),
)


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def make_customers(count: int, seed: int) -> list[Customer]:
    if not 100 <= count <= 100_000:
        raise ValueError("customers must be between 100 and 100000")
    rng = random.Random(seed)
    customers = []
    for _ in range(count):
        income_band = rng.randrange(3)
        tenure = min(1.0, rng.expovariate(3.0))
        friction = min(1.0, max(0.0, rng.gauss(0.62 - income_band * 0.12, 0.18)))
        prior = max(0.0, rng.gauss(0.05, 0.035))
        expected_loss = rng.lognormvariate(7.25, 0.42)
        renewal_probability = sigmoid(
            0.7 + 1.2 * tenure + 1.4 * friction - 5.0 * prior
        )
        customers.append(
            Customer(
                expected_loss=expected_loss,
                income_band=income_band,
                tenure=tenure,
                shopping_friction=friction,
                prior_increase=prior,
                renewed=int(rng.random() < renewal_probability),
            )
        )
    return customers


def _features(customer: Customer) -> tuple[float, ...]:
    return (1.0, customer.tenure, customer.shopping_friction, customer.prior_increase)


def fit_renewal_model(
    customers: Iterable[Customer], epochs: int = 350, learning_rate: float = 0.25
) -> tuple[float, ...]:
    rows = list(customers)
    weights = [0.0] * 4
    for _ in range(epochs):
        gradient = [0.0] * 4
        for customer in rows:
            xs = _features(customer)
            error = sigmoid(sum(w * x for w, x in zip(weights, xs))) - customer.renewed
            for index, value in enumerate(xs):
                gradient[index] += error * value
        scale = learning_rate / len(rows)
        weights = [w - scale * g for w, g in zip(weights, gradient)]
    return tuple(weights)


def predict_renewal(customer: Customer, weights: tuple[float, ...]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, _features(customer))))


def evaluate(
    customers: list[Customer], weights: tuple[float, ...], strategy: Strategy
) -> dict[str, float | str]:
    profit = harm = lapse_total = 0.0
    band_burdens = [0.0, 0.0, 0.0]
    band_counts = [0, 0, 0]

    for customer in customers:
        stay = predict_renewal(customer, weights)
        increase = (
            strategy.base_increase
            + strategy.loyalty_weight * stay
            + strategy.friction_weight * customer.shopping_friction
        )
        premium = customer.expected_loss * 1.22 * (1.0 + increase)
        lapse = sigmoid(-2.0 + 8.0 * increase - 0.8 * customer.shopping_friction)
        retained = 1.0 - lapse
        suppressed_claim = customer.expected_loss * strategy.claim_suppression
        deductible_harm = customer.expected_loss * 0.25 * strategy.deductible_shift
        margin = premium - customer.expected_loss + suppressed_claim
        profit += retained * margin

        burden = max(0.0, premium - customer.expected_loss * 1.22)
        individual_harm = burden + suppressed_claim + deductible_harm
        harm += retained * individual_harm
        band_burdens[customer.income_band] += individual_harm
        band_counts[customer.income_band] += 1
        lapse_total += lapse

    avg_burdens = [total / count for total, count in zip(band_burdens, band_counts)]
    disparity = max(avg_burdens) - min(avg_burdens)
    harm_score = (harm / len(customers)) + 2.0 * disparity
    return {
        "strategy": strategy.name,
        "profit": profit,
        "consumer_harm": harm_score,
        "lapse_rate": lapse_total / len(customers),
    }


def run_simulation(count: int = 5000, seed: int = 7) -> list[dict[str, float | str]]:
    customers = make_customers(count, seed)
    weights = fit_renewal_model(customers)
    results = [evaluate(customers, weights, strategy) for strategy in STRATEGIES]
    baseline = results[0]
    for result in results:
        result["profit_delta"] = float(result["profit"]) - float(baseline["profit"])
        harm_delta = float(result["consumer_harm"]) - float(baseline["consumer_harm"])
        result["verdict"] = (
            "BASELINE"
            if result["strategy"] == "baseline"
            else "REJECT"
            if harm_delta > 25.0
            else "REVIEW"
        )
    return results
