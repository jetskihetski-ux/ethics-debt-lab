import unittest

from ethics_debt_lab.simulation import make_customers, run_simulation


class SimulationTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(run_simulation(200, 11), run_simulation(200, 11))

    def test_rejects_harmful_strategies(self):
        results = run_simulation(500, 7)
        verdicts = {row["strategy"]: row["verdict"] for row in results}
        self.assertEqual(verdicts["baseline"], "BASELINE")
        self.assertIn("REJECT", verdicts.values())

    def test_refuses_tiny_or_massive_datasets(self):
        for count in (99, 100_001):
            with self.assertRaises(ValueError):
                make_customers(count, 1)


if __name__ == "__main__":
    unittest.main()
