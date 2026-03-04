import csv
import os

class ResultLogger:

    def __init__(self, file_path="results.csv"):

        self.file_path = file_path

        if not os.path.exists(file_path):

            with open(file_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "dataset",
                    "drift_type",
                    "attack_type",
                    "attack_ratio",
                    "accuracy",
                    "rlr"
                ])


    def log(self, dataset, drift_type, attack_type, attack_ratio, accuracy, rlr):

        with open(self.file_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                dataset,
                drift_type,
                attack_type,
                attack_ratio,
                accuracy,
                rlr
            ])