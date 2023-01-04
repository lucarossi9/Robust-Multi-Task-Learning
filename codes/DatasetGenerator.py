import numpy as np
import numpy.linalg as nla


class DataGenerator:
    """
    Generate dataset for multi-task regression
    """

    def __init__(self, low_rank_dist, sparse_dist, noise_dist, features_dist, label_noise_dist, sparsity_type,
                 noise_scale, noise_flag):

        # set the distribution of the matrices
        self.low_rank_dist = low_rank_dist
        self.sparse_dist = sparse_dist
        self.noise_dist = noise_dist
        self.features_dist = features_dist
        self.label_noise_dist = label_noise_dist

        # set the sparsity type of the matrix Gamma
        self.sparsity_type = sparsity_type

        # set the variance of the noise
        self.noise_scale = noise_scale

        # set the matrices as uninitialized
        self.sparse_matrix = None
        self.low_rank_matrix = None
        self.noise_matrix = None
        self.features_matrices = None
        self.label_matrix = None

        # flag = True if there is noise, otherwise no noise
        self.noise_flag = noise_flag

    def sparse_params(self, n_tasks, high_dim, sparsity):
        """
        Generate random sparse matrix.
        :param n_tasks: number of tasks.
        :param high_dim:  Dimensionality of the features.
        :param sparsity: Number of non-zero entries.
        :return:
        """
        if self.sparsity_type == "entrywise":
            # generate random matrix
            matrix = self.sparse_dist(size=(high_dim, n_tasks), scale=self.noise_scale)

            # generate mask for the entries to keep
            random_mask = np.full(high_dim * n_tasks, False)
            random_mask[:round(sparsity)] = True
            np.random.shuffle(random_mask)
            random_mask = random_mask.reshape((high_dim, n_tasks))

            # apply the mask
            matrix = matrix * random_mask
        else:
            # in this case the non zero entries are distributed equally in each column
            matrix = np.zeros((high_dim, n_tasks))

            # in this case sparsity is the number of non-zero entries in each column
            entries = self.sparse_dist(size=(sparsity, n_tasks), scale=self.noise_scale)
            for i in range(n_tasks):
                matrix[:sparsity, i] = entries[:, i]
                np.random.shuffle(matrix[:, i])

        self.sparse_matrix = matrix
        return

    def low_rank_params(self, n_tasks, high_dim, low_dim):
        """
        Generate random low rank matrix.
        :param n_tasks: Number of tasks.
        :param high_dim: Dimensionality of the features.
        :param low_dim: Rank of the low dimensional matrix.
        :return:
        """

        # generate random matrix
        matrix = self.low_rank_dist(size=(high_dim, high_dim), scale=self.noise_scale)

        # select top-r eigenvectors
        U, _, _ = nla.svd(matrix)
        subspace = U[:, :low_dim]

        # generate low rank matrix of rank k
        alpha = self.low_rank_dist(size=(low_dim, n_tasks), scale=self.noise_scale)
        self.low_rank_matrix = subspace @ alpha
        return

    def noise_params(self, task_size, n_tasks, flag):
        """
        Generate random noise matrix.
        :param flag: if there is noise or not.
        :param task_size: number of samples per task.
        :param n_tasks: number of tasks.
        :return:
        """
        if flag == True:
            # generate from random normal
            self.noise_matrix = self.noise_dist(size=(task_size, n_tasks), scale=self.noise_scale * 0.1)
        else:
            # set the matrix to zero
            self.noise_matrix = np.zeros((task_size, n_tasks))
        return

    def features_params(self, task_size, high_dim, n_tasks):
        """
        Generate random matrices of features.
        :param task_size: number of samples per task.
        :param high_dim: dimensionality of the features.
        :param n_tasks: number of tasks
        :return:
        """
        # generate a list of random matrices
        self.features_matrices = [self.noise_dist(size=(task_size, high_dim), scale=self.noise_scale) for i in
                                  range(n_tasks)]
        return

    def labels_params(self):
        """
        Generate labels.
        :return: labels of the multitask linear regression.
        """
        sparse_mat = self.sparse_matrix
        low_rank_mat = self.low_rank_matrix
        noise_mat = self.noise_matrix
        features_mat = self.features_matrices

        if (sparse_mat is None) or (low_rank_mat is None) or (noise_mat is None) or (features_mat is None):
            raise ValueError("Call other methods before calling labels_params")

        task_size = noise_mat.shape[0]
        n_tasks = noise_mat.shape[1]
        labels = np.zeros((task_size, n_tasks))

        # generate the matrix column by column one for each task
        for i in range(n_tasks):
            labels[:, i] = features_mat[i] @ (sparse_mat[:, i] + low_rank_mat[:, i]) + noise_mat[:, i]
        self.label_matrix = labels

        return

    def return_dataset(self, n_tasks, high_dim, low_dim, task_size, sparsity):
        """
        The functions returns the synthetic dataset.
        :param n_tasks: number of tasks.
        :param high_dim: dimensionality of the features.
        :param low_dim: rank of the low-rank matrix.
        :param task_size: number of samples per task.
        :param sparsity: the type of sparsity for the sparse matrix
        :return: A dictionnary containing the matrices
         [sparse_mat, low_rank_mat, noise_mat, features_mat, label_mat]
        """

        # generate the matrices one by one
        self.sparse_params(n_tasks, high_dim, sparsity)
        self.low_rank_params(n_tasks, high_dim, low_dim)
        self.noise_params(task_size, n_tasks, self.noise_flag)
        self.features_params(task_size, high_dim, n_tasks)
        self.labels_params()

        # set them as attributes
        sparse_mat = self.sparse_matrix
        low_rank_mat = self.low_rank_matrix
        noise_mat = self.noise_matrix
        features_mat = self.features_matrices
        label_mat = self.label_matrix

        return {"sparse": sparse_mat, "low_rank": low_rank_mat,
                "noise": noise_mat, "features": features_mat, "labels": label_mat}
