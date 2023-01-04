import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()


def plot_normalized_ratio(path):
    """
    Plot normalized ratio from the folder with the path given.
    :param path: the path containing the pickle files.
    :return:
    """

    ratio_of_recovery = []
    for i in range(7):
        with open(path + 'run_' + str(i) + '.pkl', 'rb') as f:
            # load the file
            res = pkl.load(f)

            # extract the true matrices
            true_sparse = res["dataset"]["sparse"]
            true_low_rank = res["dataset"]["low_rank"]

            # extract the predicted matrices
            predicted_sparse = res["results"]["sparses"][-1]
            predicted_low_rank = res["results"]["low_ranks"][-1]

            # compute and append the ratio of recovery
            ratio_of_recovery.append(
                (np.linalg.norm(predicted_low_rank - true_low_rank, ord="fro") ** 2 +
                 np.linalg.norm(predicted_sparse - true_sparse, ord="fro") ** 2) / (
                        np.linalg.norm(true_sparse, ord="fro") ** 2 + np.linalg.norm(true_low_rank, ord="fro") ** 2)
            )

    xs = [50 * 2 ** i for i in range(7)]
    df_ratio_recovery = pd.DataFrame({"number of tasks": xs, "normalized ratio": ratio_of_recovery})
    sns.set_style("darkgrid")
    ax = sns.lineplot(x="number of tasks", y="normalized ratio", data=df_ratio_recovery, marker="o")
    plt.legend(["low-rank+sparse"])
    ax.set(xscale='log')
    ax.set(xticks=xs)
    ax.set(xticklabels=xs)
    plt.show()
    return


def plot_normalized_distance(path):
    """
    The function plots the normalized distance for our model and for the baseline
    when the number of tasks increases
    :param path: the path of the folder where the data is saved
    :return:
    """
    xs = [50 * 2 ** i for i in range(7)]
    normalized_distance_base = []
    normalized_distance = []
    for i in range(7):
        # find normalized distance for the run
        with open('Results/plots-1-2/run_' + str(i) + '.pkl', 'rb') as f:
            res = pkl.load(f)
            true_sparse = res["dataset"]["sparse"]
            true_low_rank = res["dataset"]["low_rank"]
            true_B = true_sparse + true_low_rank
            predicted_sparse = res["results"]["sparses"][-1]
            predicted_low_rank = res["results"]["low_ranks"][-1]
            estimated_B = predicted_sparse + predicted_low_rank
            normalized_distance.append(np.linalg.norm(true_B - estimated_B, ord="fro") / np.sqrt(xs[i]))
        with open('Results/plots-1-2/run_' + str(i) + '_baseline.pkl', 'rb') as f:
            # find normalized distance for the run
            res = pkl.load(f)
            estimated_B = res["matrix"]
            true_B = true_sparse + true_low_rank
            normalized_distance_base.append(np.linalg.norm(true_B - estimated_B, ord="fro") / np.sqrt(xs[i]))

    # plot the normalized distance
    df_normalized_distance = pd.DataFrame({"number of tasks": xs, "low-rank+sparse": normalized_distance,
                                           "single": normalized_distance_base})
    new_df = pd.melt(df_normalized_distance, ['number of tasks'])
    sns.lineplot(data=new_df, x="number of tasks", y="value", hue="variable", markers=["o", "o"], style="variable")
    plt.ylabel("Normalized Frobenius distance")
    plt.legend(bbox_to_anchor=(0.65, 0.90), loc='upper left', borderaxespad=0)
    plt.show()
    return


def plot_parameter_dependence(path):
    """
    Plot the dependence of the Frobenius Errors squared with respect to r and s.
    :param path: the path of the folder where the data is saved.
    :return:
    """

    # compute the Frobenius errors squared for the low-rank
    errors_rank = []
    for i in range(1, 6):
        with open(path + 'run_' + str(i) + '_low_rank.pkl', 'rb') as f:
            res = pkl.load(f)
            true_sparse = res["dataset"]["sparse"]
            true_low_rank = res["dataset"]["low_rank"]
            predicted_sparse = res["results"]["sparses"][-1]
            predicted_low_rank = res["results"]["low_ranks"][-1]
            errors_rank.append(
                np.linalg.norm(predicted_low_rank - true_low_rank, ord="fro") ** 2 +
                np.linalg.norm(predicted_sparse - true_sparse, ord="fro") ** 2
            )
    x = np.array([10, 15, 20, 25, 30])

    # compute errors for Frobenius errors squared for the sparse matrix.
    errors_sparse = []
    for i in range(1, 6):
        with open(path + 'run_' + str(i) + '_sparse.pkl', 'rb') as f:
            res = pkl.load(f)
            true_sparse = res["dataset"]["sparse"]
            true_low_rank = res["dataset"]["low_rank"]
            predicted_sparse = res["results"]["sparses"][-1]
            predicted_low_rank = res["results"]["low_ranks"][-1]
            errors_sparse.append(
                np.linalg.norm(predicted_low_rank - true_low_rank, ord="fro") ** 2 +
                np.linalg.norm(predicted_sparse - true_sparse, ord="fro") ** 2
            )

    x_rank = np.array([10, 15, 20, 25, 30])
    y_rank = np.array(errors_rank)
    # find line of best fit
    a_rank, b_rank = np.polyfit(x_rank, y_rank, 1)

    x_sparse = np.array([1200, 1400, 1600, 1800, 2000])
    y_sparse = np.array(errors_sparse)
    # find line of best fit
    a_sparse, b_sparse = np.polyfit(x_sparse, y_sparse, 1)

    # plot as a subplot both the errors as a scatter plot and the best line fitting the points
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_rank, y_rank)
    ax2.scatter(x_sparse, y_sparse)
    ax1.plot(x_rank, a_rank * x_rank + b_rank)
    ax2.plot(x_sparse, a_sparse * x_sparse + b_sparse)
    ax1.set_title("Errors vs rank")
    ax2.set_title("Errors vs sparsity")
    ax1.set_xlabel("rank")
    ax2.set_xlabel("sparsity")
    ax1.set_ylabel("Frobenius errors squared")
    ax2.set_ylabel("Frobenius errors squared")
    plt.show()
    return


plot_normalized_ratio("Results/plots-1-2/")
plot_normalized_distance("Results/plots-1-2/")
plot_parameter_dependence("Results/plot-3/")
