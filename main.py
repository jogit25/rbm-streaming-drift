import numpy as np
import torch

from models.robust_rbm import RobustSupervisedRBM
from drift.drift_detector import RRBMDriftDetector
from evaluation.prequential import PrequentialEvaluator

if __name__ == "__main__":

    n_features = 10
    n_classes = 2
    hidden_units = 5

    rbm = RobustSupervisedRBM(n_features, hidden_units, n_classes)
    drift_detector = RRBMDriftDetector()
    evaluator = PrequentialEvaluator(drift_detector)

    print("Starting modular test...")

    for _ in range(200):
        x = np.random.rand(n_features)
        y = np.random.randint(0, n_classes)

        x_dict = {f"f{i}": float(x[i]) for i in range(n_features)}
        v_tensor = torch.tensor([x], dtype=torch.float32)

        evaluator.process_instance(x_dict, y, v_tensor, rbm)

    print("Final Accuracy:", evaluator.get_accuracy())