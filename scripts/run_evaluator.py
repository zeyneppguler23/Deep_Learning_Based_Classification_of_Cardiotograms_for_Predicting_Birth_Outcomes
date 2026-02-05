import sys
import pathlib
import numpy as np

# ensure project root is on sys.path so `src` imports resolve
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.utils.evaluator import Evaluator


def main():
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=100)
    # create probabilities somewhat correlated with labels
    y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, 100)
    y_prob = np.clip(y_prob, 0, 1)

    thresholds = [0.4, 0.5, 0.6]
    ev = Evaluator(thresholds)
    results = ev.evaluate_with_print(y_true, y_prob, verbose=True)

    print("\nReturned keys:", list(results.keys()))
    print("Threshold results:")
    for thr, vals in results['thresholds'].items():
        print(f"  {thr}: {vals}")


if __name__ == '__main__':
    main()
