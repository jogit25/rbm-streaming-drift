import pandas as pd
import matplotlib.pyplot as plt


def plot_results(file_path="results.csv"):

    df = pd.read_csv(file_path)

    drift_types = df["drift_type"].unique()

    for drift in drift_types:

        subset = df[df["drift_type"] == drift]

        plt.figure()

        for attack in subset["attack_type"].unique():

            attack_data = subset[subset["attack_type"] == attack]

            plt.bar(
                attack,
                attack_data["accuracy"].mean()
            )

        plt.title(f"Accuracy under {drift} drift")
        plt.ylabel("Accuracy")

        plt.show()