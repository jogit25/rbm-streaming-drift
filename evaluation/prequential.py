import torch
import torch.nn.functional as F
from river import tree, metrics

class PrequentialEvaluator:
    def __init__(self, drift_detector):
        self.classifier = tree.HoeffdingAdaptiveTreeClassifier()
        self.metric = metrics.Accuracy()
        self.drift_detector = drift_detector
        self.classifier_error_series = []

    def process_instance(self, x_dict, y_true, v_tensor, rbm_model):

        y_pred = self.classifier.predict_one(x_dict)
        if y_pred is None:
            y_pred = y_true

        self.metric.update(y_true, y_pred)

        error = 0 if y_pred == y_true else 1
        self.classifier_error_series.append(error)

        v_recon, _ = rbm_model.contrastive_divergence(
            v_tensor,
            F.one_hot(torch.tensor([y_true]), num_classes=rbm_model.Z).float()
        )

        recon_error = torch.mean((v_tensor - v_recon) ** 2).item()

        drift_flag = self.drift_detector.update_and_check(
            recon_error,
            self.classifier_error_series
        )

        if drift_flag:
            self.classifier = tree.HoeffdingAdaptiveTreeClassifier()

        self.classifier.learn_one(x_dict, y_true)

    def get_accuracy(self):
        return self.metric.get()