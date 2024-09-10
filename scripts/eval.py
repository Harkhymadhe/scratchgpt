# -*- coding: utf-8 -*-

import os
import torch
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

__all__ = [
    "extract_metrics",
    "plot_metric",
]


def extract_metrics(timestamp, subfolder):

    folder = f"../artefacts/models/{timestamp}/{subfolder}"
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    file_paths.sort()

    state_dict_path, _, model_config_path = file_paths
    state_dict = torch.load(state_dict_path, map_location="cpu")
    metric_reports = {
        k: v for k, v in state_dict.items() if k in ["loss", "accuracy", "perplexity"]
    }

    return metric_reports


def plot_metric(metric_report, model_name, metric="loss", fig_size=None):
    allowed_metrics = ["loss", "perplexity", "accuracy", "all"]

    assert (
        metric in allowed_metrics
    ), "metric must be one of ['loss', 'accuracy', 'perplexity', 'all']"

    if fig_size is None:
        fig_size = (10, 5) if metric != "all" else (10, 20)

    if metric != "all":
        train_metric_report, test_metric_report = metric_report[metric]
        num_epochs = len(train_metric_report)

        plt.figure(figsize=fig_size)

        plt.plot(
            range(1, 1 + num_epochs),
            train_metric_report,
            "-o",
            color="blue",
            label=f"Train {metric}",
        )
        plt.plot(
            range(1, 1 + num_epochs),
            test_metric_report,
            "-x",
            color="green",
            label=f"Test {metric}",
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())

        plt.title(f"Evolution of {metric} over {num_epochs} epochs")
    else:
        fig, ax = plt.subplots(figsize=fig_size, nrows=3, ncols=1, sharex=False)

        for i, metric_ in enumerate(allowed_metrics):
            if metric_ == "all":
                continue

            train_metric_report, test_metric_report = metric_report[metric_]
            num_epochs = len(train_metric_report)

            ax[i].plot(
                range(1, 1 + num_epochs),
                train_metric_report,
                "-o",
                color="blue",
                label=f"Train {metric_}",
            )
            ax[i].plot(
                range(1, 1 + num_epochs),
                test_metric_report,
                "-o",
                color="green",
                label=f"Train {metric_}",
            )

            ax[i].set_title(f"Evolution of {metric_} over {num_epochs} epochs")
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel(metric_.capitalize())

    plt.legend()
    path = f"../artefacts/metrics/{model_name}"
    if not os.path.exists(path):
        os.makedirs(f"../artefacts/metrics/{model_name}")

    plt.savefig(f"../artefacts/metrics/{model_name}/{metric}.png", dpi=300)
    plt.show()
    plt.close("all")

    return


if __name__ == "__main__":
    choice = None

    while choice not in ["orwell", "poe"]:
        choice = input("Input one of 'orwell' or 'poe': ").lower()

    if choice == "orwell":
        timestamp = "20240827"
        subfolder = 1
        tokenizer_path = "george_orwell.json"
    else:
        timestamp = "20240904"
        subfolder = 1
        tokenizer_path = "edgar_alan_poe.json"

    model_name = tokenizer_path.split(".")[0]

    metric = "loss"
    from pprint import pprint

    metric_reports = extract_metrics(
        timestamp=timestamp,
        subfolder=subfolder,
    )

    pprint(metric_reports)

    import pandas as pd

    df = pd.DataFrame(
        data={
            f"train_loss": [round(m, 3) for m in metric_reports["loss"][0]],
            f"test_loss": [round(m, 3) for m in metric_reports["loss"][1]],
            f"train_accuracy": [round(100 * m, 3) for m in metric_reports["accuracy"][0]],
            f"test_accuracy": [round(100 * m, 3) for m in metric_reports["accuracy"][1]],
            f"train_perplexity": [round(m, 3) for m in metric_reports["perplexity"][0]],
            f"test_perplexity": [round(m, 3) for m in metric_reports["perplexity"][1]],
        },
    )

    metrics = ["loss", "perplexity", "accuracy"]
    for metric in metrics:
        plot_metric(metric_report=metric_reports,  model_name=model_name, metric=metric)

    df.to_excel(f"../artefacts/metrics/{model_name}/metrics.xlsx", index=False)

    print(df)
