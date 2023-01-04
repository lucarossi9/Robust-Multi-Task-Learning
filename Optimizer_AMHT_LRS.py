import numpy as np
import numpy.linalg as la


class Optimizer_AMHT_LRS:
    """ Contains the optimizer AMHT-LRS """

    def __init__(self, features=None, labels=None):
        self.features = features
        self.labels = labels

    def compute_loss(self, sparse_mat, low_rank_mat, features, labels):
        """
        Compute the MSE loss for the linear regression task.
        :param sparse_mat: The sparse matrix.
        :param low_rank_mat: The low rank matrix.
        :param features: The features.
        :param labels: The labels.
        :return: The MSE loss.
        """
        n_tasks = labels.shape[1]
        loss = 0
        for i in range(n_tasks):
            loss += np.linalg.norm(labels[:, i] - features[i] @ (low_rank_mat[:, i] + sparse_mat[:, i])) ** 2
        return loss / 2

    def hard_thresholding(self, v, Delta):
        """
        Performs the hard thresholding of the vector v with parameter Delta.
        :param v: The vector to make sparse.
        :param Delta: The threshold.
        :return: The thresholded vector.
        """
        return (np.abs(v) > Delta) * v

    def optimize_sparse_vector(self, i, v, b, T_iters, alpha, beta, gamma, sparsity, features, labels, lr, c_1=1 / 4):
        """
        Solve the first sub-problem consisting in optimizing the sparse vector b.
        :param i: index of the vector to optimize: b is the i-th column of the sparse matrix.
        :param v: the vector v= Uw^(i) from the previous iteration.
        :param b: initialization point.
        :param T_iters: number of iterations.
        :param alpha: positive parameter.
        :param beta: positive parameter.
        :param gamma: positive parameter.
        :param sparsity: The sparsity of the vector.
        :param features: the features.
        :param labels: The labels.
        :param c_1: parameter such that 0<c_1<1/2
        :return: optimized vector b.
        """
        m = features[i].shape[0]
        for j in range(T_iters):

            # gradient step
            c = b - lr * features[i].T @ (
                    features[i] @ b + features[i] @ v - labels[:, i])
            Delta = alpha + c_1 * (gamma + beta / np.sqrt(sparsity))
            b = self.hard_thresholding(c, Delta)
            gamma = 2 * c_1 * gamma + 2 * (alpha + c_1 * beta / np.sqrt(sparsity))

        return b

    def eigs(self, M):

        eigenValues, eigenVectors = la.eig(M)

        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]

        return eigenValues, eigenVectors

    def AMHT_LRS(self, n_iters, sparsity, gamma, B, U, W, sparse_mat, features_train, features_test, labels_train,
                 labels_test, c_3, c_4, c_5, lr, eps=1e-2, MOM=False):
        """
        The function applies the AMHT-LRS algorithm to optimize the multi-task loss function.
        :param n_iters: The number of iterations of the algorithm.
        :param sparsity: Upper bound of non_zero entries of each column of sparse_mat.
        :param gamma: Parameter giving upper bound on ||Vec(sparse_mat) -Vec(true_sparse)||_inf
        :param B: Parameter giving upper bound between ||U-true_U||_F
        :param U: The initial iterate of the orthonormal matrix U.
        :param W: The initial iterate of the orthonormal matrix V.
        :param sparse_mat: The sparse matrix.
        :param features_train: The features of the train set.
        :param features_test: The features of the val/test set.
        :param labels_train: The labels of the train set.
        :param labels_test: The labels of the val/test set.
        :param c_3: Parameter for the update of gamma
        :param c_4: Parameter for optimize_sparse_vector method.
        :param c_5: Parameter for optimize_sparse_vector method.
        :param eps: Tolerance parameter.
        :param MOM: Flag, MOM=True if we initialize with the MOM, false otherwise.
        :return: A series of losses, and of matrices U, W, sparse_mat
        """
        n_tasks = len(features_train)
        task_size = features_train[0].shape[0]
        d = features_train[0].shape[1]
        # use the method of moments for initialization
        if MOM:
            M = np.zeros(shape=(d, d))
            for i in range(n_tasks):
                X = features_train[i]
                y = labels_train[:, i]
                scaled_X = (X.T * y).T
                M += scaled_X.T @ scaled_X
            # M = 1 / float(task_size*n_tasks) * M
            eigVals, eigVecs = self.eigs(M)
            U = eigVecs[:, :U.shape[1]]

        # initialization
        Ws = []
        Us = []
        sparses = []
        losses = []
        loss = self.compute_loss(sparse_mat, U @ W, features_test, labels_test)
        losses.append(loss)
        sparses.append(sparse_mat)
        Ws.append(W)
        Us.append(U)
        print(f"---iter={0}--- loss={loss}----")
        for l in range(n_iters):
            T_iters = int(round(2 * l * np.log(gamma / eps)))
            for i in range(n_tasks):
                alpha = (1 / 2) * c_4 * B / np.sqrt(sparsity)
                beta = (1 / 2) * c_5 * B
                # update sparse matrix
                sparse_mat[:, i] = self.optimize_sparse_vector(i, U @ W[:, i], sparse_mat[:, i], T_iters, alpha,
                                                               beta, gamma,
                                                               sparsity, features_train, labels_train, lr, 1 / 4)
                # update W with closed form solution
                a = (features_train[i] @ U).T @ (features_train[i] @ U)
                b = (features_train[i] @ U).T @ (labels_train[:, i] - features_train[i] @ sparse_mat[:, i])
                W[:, i] = np.linalg.solve(a, b)

            d = features_train[0].shape[1]
            r = W.shape[0]
            A = np.zeros((d * r, d * r))
            for i in range(n_tasks):
                tmp = np.zeros((d, d))
                for j in range(task_size):
                    tmp += np.dot(features_train[i][j, :].reshape(-1, 1), features_train[i][j, :].reshape(1, -1))
                w = W[:, i]
                A = A + np.kron(np.dot(w.reshape(-1, 1), w.reshape(1, -1)), tmp)

            # compute V and update U
            V = np.zeros((d, r))
            for i in range(n_tasks):
                V += features_train[i].T @ (labels_train[:, i] -
                                            features_train[i] @ sparse_mat[:, i]).reshape(-1, 1) @ (
                         W[:, i].T.reshape(1, -1))
            U = np.reshape(np.linalg.solve(A, V.reshape(-1, 1)), (d, r))
            U, null = np.linalg.qr(U)
            gamma = c_3 * eps * B

            # compute statistics
            low_rank_mat = U @ W
            loss = self.compute_loss(sparse_mat, low_rank_mat, features_test, labels_test)
            print(f"---iter={l + 1}--- loss={loss}----")
            sparses.append(sparse_mat)
            Ws.append(W)
            Us.append(U)
            losses.append(loss)

        return {"losses": losses, "sparses": sparses, "Us": Us, "Ws": Ws}
