# Ethics Debt Lab

An intentionally dystopian insurance-ML red-team simulator.

It answers a narrow question: **how profitable can a technically legal but
consumer-hostile insurance strategy look before its harm becomes impossible to
ignore?**

The project generates synthetic policyholders, estimates renewal propensity,
applies four "dark" strategies, and produces a profit-versus-harm audit. It is
designed for ethics workshops, model-risk reviews, and policy discussions—not
for underwriting, pricing, claims decisions, or use with real people.

## Simulated strategies

- **Loyalty tax** — larger renewal increases for customers predicted to stay.
- **Complexity rent** — higher increases where comparison-shopping friction is high.
- **Claims friction** — simulated administrative delay that suppresses small claims.
- **Coverage shrink** — a less visible deductible increase instead of a headline premium increase.

Protected traits are deliberately excluded. The simulator operates on synthetic,
aggregate cohorts and refuses external customer files.

## Quick start

```bash
python -m ethics_debt_lab --seed 7 --customers 5000
```

Example output:

```text
strategy          profit_delta   consumer_harm   lapse_rate   verdict
baseline          $0             0.0             ...
loyalty_tax       ...            ...             ...          REJECT
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## What the model is

Renewal probability is estimated with a tiny logistic model trained by gradient
descent on synthetic behavioral signals. The "optimizer" tests a fixed menu of
predefined scenarios. It does not recommend a price for an identifiable person.

The harm score combines:

- premium burden above expected loss;
- simulated unpaid claim value;
- deductible surprise;
- disparate impact across synthetic income bands.

Any strategy that improves profit while breaching a harm guardrail receives a
`REJECT` verdict. This is the point of the project: profit alone is an
incomplete and often dangerous objective.

## Safety boundary

Do not connect this code to customer records, quote systems, claims workflows,
or production decisioning. Laws vary by jurisdiction, and "legal" is not a
claim made by this repository. The scenarios are fictional provocations for
review and education.

## License

MIT. See [LICENSE](LICENSE).
