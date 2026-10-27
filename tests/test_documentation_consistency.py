"""Check README notation and the explanatory N* labels against the estimators."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationConsistency(unittest.TestCase):
    def test_readme_uses_estimator_names(self):
        readme = (ROOT / "README.md").read_text()
        estimator = (ROOT / "fractalsig/features/fractal_math.py").read_text()

        self.assertIn("Hurst exponent **H** (rescaled range)", readme)
        self.assertIn("Hölder exponent **S**\n(return probability)", readme)
        self.assertIn("``calc_H`` -- Hurst exponent by rescaled-range", estimator)
        self.assertIn("``calc_S`` -- Hoelder (self-similarity) exponent from the return probability", estimator)

    def test_n_star_labels_match_reference_output_and_generator(self):
        readme = (ROOT / "README.md").read_text()
        stats = json.loads((ROOT / "reference_outputs/stats/t2_n_star_composition.json").read_text())
        generator = (ROOT / "experiments/consolidate_t0_t1_t2.py").read_text()

        self.assertIn("fBm floor: H point estimate, S lower bound", readme)
        for key, value in stats.items():
            self.assertIn("point estimate" if key.endswith("_H") else "lower bound", value["note"])
        self.assertIn('if est_key == "S" else', generator)


if __name__ == "__main__":
    unittest.main()
