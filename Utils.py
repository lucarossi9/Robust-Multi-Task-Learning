from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.transforms as transforms
import math


def loss_analysis(losses, dataset, optimizer):
    """
    The function analyzes the losses during training and the ones of the ground_truth
    :param losses: The training losses.
    :param dataset: The dataset.
    :param optimizer: The optimizer.
    :return:
    """
    # show the MSE of the true parameters (MSE different from zero if there is noise)
    print(f"MSE ground-truth=", optimizer.compute_loss(dataset["sparse"], dataset["low_rank"], dataset["features"],
                                                      dataset["labels"]))
    print(f"Minimum loss reached during training is {min(losses)}")
    plt.plot(np.arange(len(losses)), losses)
    plt.yscale("log")
    plt.title("Training losses")
    plt.xlabel("iterations")
    plt.ylabel("losses")
    plt.show()


def ratio_of_recovery_analysis(dataset, results):
    """
    The function analyses and plot the ratio of recovery
    :param dataset: The generated dataset.
    :param results: The results of the training procedure.
    :return:
    """

    # lists containing Frobenius squared distances between true and estimated matrices
    sq_dist_sparse = []
    sq_dist_low_rank = []
    n = len(results["sparses"])

    # compute the Frobenius squared distances
    for i in range(n):
        sq_dist_sparse.append(np.linalg.norm(dataset["sparse"] - results["sparses"][i], ord='fro') ** 2)
        sq_dist_low_rank.append(np.linalg.norm(dataset["low_rank"] - results["low_ranks"][i], ord='fro') ** 2)

    # compute normalized ratio
    normalized_dist = (np.array(sq_dist_low_rank) + np.array(sq_dist_sparse)) / (
            np.linalg.norm(dataset["sparse"], ord="fro") ** 2 + np.linalg.norm(dataset["low_rank"], ord="fro") ** 2)


    print("last ratio reached=", normalized_dist[-1])
    print("Minimum ratio reached=", np.min(normalized_dist))
    fig, ax = plt.subplots()
    ax.plot(np.arange(n), normalized_dist)
    ax.axhline(y=0.1, color='r', linestyle='--')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, 0.1, "{}".format(0.1), color="red", transform=trans,
            ha="right", va="center")
    plt.xlabel("number of iterations")
    plt.ylabel("ratio")
    plt.title("Ratio of recovery")
    plt.show()


def matrices_info(results, dataset, sparsity_type, low_dim):
    """
    The function prints useful information regarding the recovered matrices.
    :param results: The results of the optmization.
    :param dataset: The original dataset.
    :param sparsity_type: The sparsity type.
    :param low_dim: the rank of the low rank matrix
    :return:
    """
    n_tasks = dataset["sparse"].shape[1]

    # print the rank and the sparsity of the two estimates

    print("the rank of the recovered matrix is", np.linalg.matrix_rank(results["low_ranks"][-1]))
    if sparsity_type == "entrywise":
        print("the sparsity of the recovered matrix is",
              np.count_nonzero(results["sparses"][-1]) / (high_dim * n_tasks))
    else:
        print("the sparsity of the recovered matrix is", np.count_nonzero(results["sparses"][-1]) / (n_tasks))


def display_regularization_params(dataset):
    """
    Computes the hyperparams to use to have the guarantees of the theorem
    :param dataset: The dataset
    :return: None
    """
    n_tasks = dataset["labels"].shape[1]
    task_size = dataset["labels"].shape[0]
    high_dim = dataset["features"][0].shape[1]

    # compute k_max and sigma_max in the formula for the hyperparams
    k_max = 0
    sigma_max = 0
    for i in range(n_tasks):
        U, S, Vh = np.linalg.svd(dataset["features"][i] / np.sqrt(task_size))
        if np.max(S) > sigma_max:
            sigma_max = np.max(S)
        if max(np.linalg.norm(dataset["features"][0], axis=0)) > k_max:
            k_max = max(np.linalg.norm(dataset["features"][i] / np.sqrt(task_size), axis=0))

    # compute lambda_d and mu_d predicted by the formula in the corollary
    # we will scale them properly once and use always the scaled ones in the experiments
    print("lambda predicted", 16 * 0.1 * sigma_max * np.sqrt(task_size) * (np.sqrt(high_dim) + np.sqrt(n_tasks)))
    print("mu predicted", 16 * 0.1 * k_max * np.sqrt(task_size * np.log(high_dim * n_tasks)) +
          4 * np.max(dataset["low_rank"]) * 0.1 * np.sqrt(task_size) / np.sqrt(n_tasks * high_dim))
    return
