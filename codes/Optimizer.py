import os
os.environ["SCIPY_USE_PROPACK"] = "1"
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import scipy
from scipy.sparse.linalg import svds
import time

class Optimizers:
    """ Contains the optimizers for the multi-task linear regression problem
    We implemented:
    - Projected Gradient Descent
    - Proximal Gradient Descent
    - Frank-Wolfe Thresholding
    """

    def __init__(self, features=None, labels=None):

        self.features = features
        self.labels = labels

    def simplex_projection(self, v, s):
        """
        Performs projection onto the unit simplex, useful for Projected Gradient Descent.
        :param: v: vector to project onto the unit simplex.
        :param: s: radius of the simplex.
        :return: The projected vector.
        """

        # credits to https://gist.github.com/mblondel/6f3b7aaad90606b98f71

        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() <= s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = float(cssv[rho] - s) / (rho + 1)
        # compute the projection by thresholding v using theta
        w = np.maximum(v - theta, 0)
        return w

    def projection_L1(self, v, s):
        """
        Performs projection onto the L1 unit ball. Useful for Projected Gradient Descent.
        :param v: The vector to project.
        :param s: The radius of the L1 ball.
        :return: The projection.
        """
        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # compute the vector of absolute values
        u = np.abs(v)
        # check if v is already a solution
        if u.sum() <= s:
            # L1-norm is <= s
            return v
        # v is not already a solution: optimum lies on the boundary (norm == s)
        # project *u* on the simplex
        w = self.simplex_projection(u, s=s)
        # compute the solution to the original problem on v
        w *= np.sign(v)
        return w

    def projection_low_rank(self, low_rank_mat, radius):
        """
        Performs projection onto nuclear ball.
        :param low_rank_mat: The matrix to project.
        :param radius: The radius of the nuclear ball.
        :return: The projected matrix.
        """
        U, s, V = np.linalg.svd(low_rank_mat, full_matrices=False)
        # we project the eigenvalues onto the simplex
        s = self.simplex_projection(s, radius)
        return U.dot(np.diag(s).dot(V))

    def projection_sparse(self, sparse_mat, radius):
        """
        Performs projection onto L1 ball.
        :param sparse_mat: The matrix to project.
        :param radius: The radius of the L1 ball.
        :return: The projected matrix.
        """
        n_rows = sparse_mat.shape[0]
        n_cols = sparse_mat.shape[1]
        v = sparse_mat.reshape(-1)
        v_proj = self.projection_L1(v, radius)
        return v_proj.reshape((n_rows, n_cols))

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

    def compute_gradient(self, sparse_mat, low_rank_mat, features, labels):
        """
        Compute the gradient of the MSE loss (w.r.t. low_rank or sparse not both).
        :param sparse_mat: The sparse matrix.
        :param low_rank_mat: The low_rank matrix.
        :param features: The features.
        :param labels: The labels.
        :return: The gradient of the MSE loss.
        """
        grad = np.zeros(sparse_mat.shape)
        n_tasks = sparse_mat.shape[1]
        for i in range(n_tasks):
            grad[:, i] = -np.dot(features[i].T, (labels[:, i] - np.dot(features[i], (low_rank_mat[:, i] +
                                                                                     sparse_mat[:, i]))))
        return grad

    def cross_validation_PGD(self, radii_sparse, radii_low_rank, n_iter, step_size, low_rank_mat, sparse_mat, K=5):
        """
        Performs cross validation to select the best values of radii.
        :param radii_sparse: The radius of L1 ball.
        :param radii_low_rank: The radius of the nuclear ball.
        :param n_iter: The number of iterations of PGD.
        :param step_size: The gamma of gradient descent.
        :param low_rank_mat: The low-rank matrix, starting point.
        :param sparse_mat: The sparse matrix, starting point.
        :param K: the number of splits.
        :return: The best radii and the results when fitting the model again with these radii.
        """

        # use KFold cross validation
        kf = KFold(n_splits=K)
        X = self.features
        y = self.labels

        # the matrix will contain the results after the cross-validation
        result_matrix = np.zeros((len(radii_sparse), len(radii_low_rank)))
        n_tasks = y.shape[1]
        for train_index, test_index in kf.split(X[0]):
            X_train = []
            X_val = []
            # split for cross-validation
            for i in range(n_tasks):
                X_train.append(X[i][train_index])
                X_val.append(X[i][test_index])

            y_train, y_val = y[train_index], y[test_index]

            for i, radius_sparse in enumerate(radii_sparse):
                for j, radius_low_rank in enumerate(radii_low_rank):
                    results = self.projected_gradient_descent(step_size, radius_sparse, radius_low_rank, n_iter,
                                                              low_rank_mat, sparse_mat, X_train, y_train, X_val, y_val)
                    result_matrix[i, j] += min(results["losses"])  # early stopping

        # average of the K folds
        result_matrix = result_matrix / K

        # extract best radius sparse and low_rank
        max_pos = np.unravel_index(result_matrix.argmax(), result_matrix.shape)
        best_radius_sparse = radii_sparse[max_pos[0]]
        best_radius_low_rank = radii_low_rank[max_pos[1]]

        # rerun PGD with found parameters
        results = self.projected_gradient_descent(step_size, best_radius_sparse, best_radius_low_rank, n_iter,
                                                  low_rank_mat, sparse_mat, self.train_features, self.train_labels,
                                                  self.train_features, self.train_labels)
        return best_radius_sparse, best_radius_low_rank, results

    def projected_gradient_descent(self, step_size, radius_sparse, radius_low_rank, n_iter, low_rank_mat, sparse_mat,
                                   features_train, labels_train, features_val, labels_val):
        """
        Optimize using projected gradient descent.
        :param step_size: The gamma of gradient descent.
        :param radius_sparse: The radius of L1 ball.
        :param radius_low_rank: The radius of the nuclear ball.
        :param n_iter: The number of iterations of PGD.
        :param low_rank_mat: The low-rank matrix, starting point.
        :param sparse_mat: The sparse matrix, starting point.
        :param features_train: The features of the training set.
        :param labels_train: The labels of the training set.
        :param features_val: The features of the validation set.
        :param labels_val: The labels of the validation set.
        :return: The iterates of matrices and the iterates of losses.
        """
        losses = []
        low_ranks = []
        sparses = []
        losses.append(self.compute_loss(sparse_mat, low_rank_mat, features_val, labels_val))
        low_ranks.append(low_rank_mat)
        sparses.append(sparse_mat)
        for i in range(n_iter):
            # gradient step
            grad = self.compute_gradient(sparse_mat, low_rank_mat, features_train, labels_train)
            low_rank_mat = low_rank_mat - step_size * grad
            sparse_mat = sparse_mat - step_size * grad

            # projection
            low_rank_mat = self.projection_low_rank(low_rank_mat, radius_low_rank)
            sparse_mat = self.projection_sparse(sparse_mat, radius_sparse)

            # compute losses
            loss = self.compute_loss(sparse_mat, low_rank_mat, features_val, labels_val)
            # print(f"iteration {i}, loss={loss}")
            losses.append(loss)
            low_ranks.append(low_rank_mat)
            sparses.append(sparse_mat)
        return {"low_ranks": low_ranks, "sparses": sparses, "losses": losses}

    def soft_thresholding(self, mat, param):
        """
        Applies soft thresholding operator on matrix mat with parameter param.
        :param mat: The matrix to which we apply soft-thresholding.
        :param param: The parameter of the operator.
        :return: The thresholded matrix
        """
        new_mat = np.sign(mat) * np.maximum(np.abs(mat) - param, 0)
        return new_mat

    def soft_thresholding_eigs(self, mat, param):
        """
        Applies soft thresholding operator to the eigenvalues of the matrix mat with parameter param.
        :param mat: The matrix to which we apply soft-thresholding.
        :param param: The parameter of the operator.
        :return: The thresholded matrix
        """
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        s = self.soft_thresholding(s, param)
        return U @ np.diag(s) @ Vh

    def prox_sol_l12(self, mat, param):
        """
        NOT USED
        The functions finds the solution to the l12 proximal problem.
        :param mat: the matrix to which we apply the proximal map.
        :param param: The parameter of the map.
        :return: The solution to the prox map--> argmin (1/2)||mat-X|| + param*||X||_{1,2}
        """
        result = np.zeros(mat.shape)

        # sort the entries of the rows of the matrix
        sorted_mat = np.zeros(mat.shape)

        # mat indexed by (s, k)
        for i in range(mat.shape[0]):
            sorted_mat[i, :] = np.sort(mat[i, :])[::-1]  # sort in descending order each row

        Ks = np.zeros(mat.shape[0])
        for s in range(mat.shape[0]):
            for k in range(mat.shape[1] - 1):
                # k represents Ks[s]
                if (param * (np.sum(sorted_mat[s, :k + 1] - (k + 1) * sorted_mat[s, k])) < sorted_mat[s, k]) & (
                        param * (np.sum(sorted_mat[s, :k + 2]) - (k + 2) * sorted_mat[s, k]) >= sorted_mat[s, k]):
                    Ks[s] = k
                    break
        Ks = Ks.astype(int)
        for s in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                result[s, k] = (mat[s, k] / np.abs(mat[s, k])) * \
                               np.maximum(np.abs(mat[s, k]) - (
                                       param / (1 + param * (Ks[s]))) * np.sum(sorted_mat[s, :Ks[s] + 1]), 0)
        return result

    def cross_validation_proximal(self, lambdas_sparse, lambdas_low_rank, n_iter, step_size, low_rank_mat, sparse_mat,
                                  K=5):
        """
        Performs cross validation to select the best values of the regularizations parameters
        :param lambdas_sparse: The regularization parameter for the sparse matrix.
        :param lambdas_low_rank: The regularization parameter for the low-rank matrix.
        :param n_iter: The number of iterations of PGD.
        :param step_size: The gamma of gradient descent.
        :param low_rank_mat: The low-rank matrix.
        :param sparse_mat: The sparse matrix.
        :param K: The number of splits.
        :return: The best parameters and the results of the fit.
        """

        # performs cross validation to select the best reguarization parameters
        kf = KFold(n_splits=K)
        X = self.features
        y = self.labels
        result_matrix = np.zeros((len(lambdas_sparse), len(lambdas_low_rank)))
        n_tasks = y.shape[1]
        for train_index, test_index in kf.split(X[0]):

            X_train = []
            X_val = []
            # split for cross-validation
            for i in range(n_tasks):
                X_train.append(X[i][train_index])
                X_val.append(X[i][test_index])

            y_train, y_val = y[train_index], y[test_index]

            for i, lambda_sparse in enumerate(lambdas_sparse):
                for j, lambda_low_rank in enumerate(lambdas_low_rank):
                    results = self.proximal_method(step_size, lambda_sparse, lambda_low_rank, n_iter,
                                                   low_rank_mat, sparse_mat, X_train, y_train,
                                                   X_val, y_val)
                    result_matrix[i, j] += min(results["losses"])  # way of doing early stopping

        # average of the K folds
        result_matrix = result_matrix / K

        # extract best radius sparse and low_rank
        max_pos = np.unravel_index(result_matrix.argmax(), result_matrix.shape)
        best_lambda_sparse = lambdas_sparse[max_pos[0]]
        best_lambda_low_rank = lambdas_low_rank[max_pos[1]]

        # rerun proximal method with found parameters
        results = self.proximal_method(step_size, best_lambda_sparse, best_lambda_low_rank, n_iter,
                                       low_rank_mat, sparse_mat, self.features, self.labels,
                                       self.features, self.labels)
        return best_lambda_sparse, best_lambda_low_rank, results

    def proximal_method(self, step_size, lambda_sparse, lambda_low_rank, n_iter, low_rank_mat, sparse_mat,
                        features_train, labels_train, features_val, labels_val,
                        method_sparse="l1"):
        """
        Optimizes the regularized MSE using proximal methods.
        :param step_size: The step size for proximal gradient descent.
        :param lambda_sparse: The regularization parameter for the sparsity.
        :param lambda_low_rank: The regularization parameter for the low rank part.
        :param n_iter: The number of iterations of proximal method.
        :param low_rank_mat: The low-rank matrix.
        :param sparse_mat: The sparse matrix.
        :param features_train: The features of the training set.
        :param labels_train: The labels of the training set.
        :param features_val: The features of the validation set.
        :param labels_val: The labels of the validation set.
        :param method_sparse: Which regularization do we have for the sparse part (apply different proximal solution).
        :return: The iterates of matrices and the iterates of losses.
        """

        # elements to save
        losses = []
        low_ranks = []
        sparses = []
        losses.append(self.compute_loss(sparse_mat, low_rank_mat, features_val, labels_val))
        low_ranks.append(low_rank_mat)
        sparses.append(sparse_mat)
        for i in range(n_iter):
            grad = self.compute_gradient(sparse_mat, low_rank_mat, features_train, labels_train)
            tmp_sparse = sparse_mat - step_size * grad
            tmp_low_rank = low_rank_mat - step_size * grad
            # solve proximal problem
            if method_sparse == "l1":
                sparse_mat = self.soft_thresholding(tmp_sparse, lambda_sparse * step_size)
            else:
                sparse_mat = self.prox_sol_l12(tmp_sparse, lambda_sparse * step_size)

            low_rank_mat = self.soft_thresholding_eigs(tmp_low_rank, lambda_low_rank * step_size)

            if i % 1000 == 0:
                loss = self.compute_loss(sparse_mat, low_rank_mat, features_val, labels_val)
                if method_sparse == "l1":
                    loss = loss + lambda_low_rank * np.linalg.norm(low_rank_mat, "nuc") + lambda_sparse * np.sum(
                        np.abs(sparse_mat))
                else:
                    # we have the L12 regularizer and the loss is therefore different
                    sum = 0
                    loss = loss + lambda_low_rank * np.linalg.norm(low_rank_mat, "nuc")
                    for h in range(sparse_mat.shape[1]):
                        sum += np.linalg.norm(sparse_mat[:, h], ord=1) ** 2
                    loss += np.sqrt(sum) * lambda_sparse

                print(f"iteration {i}, loss={loss}")
                losses.append(loss)
                low_ranks.append(low_rank_mat)
                sparses.append(sparse_mat)
        return {"low_ranks": low_ranks, "sparses": sparses, "losses": losses}

    def FW_T(self, sparse, low_rank, lambda_sparse, lambda_low_rank, features, labels, n_iter, lr):
        """
        The method optimizes the objective function using the Frank-Wolfe Thresholding Algorithm.
        :param sparse: The sparse matrix.
        :param low_rank: The low rank matrix.
        :param lambda_sparse: The regularization for the sparse matrix.
        :param lambda_low_rank: The regularization for the low_rank matrix.
        :param features: The feature matrices.
        :param labels: The labels of the matrices.
        :n_iter: The maximum number of iterations.
        :lr: The learning rate.
        :return: The iterates of matrices and the iterates of losses.
        """

        # initialization
        t_l = 0
        t_s = 0
        U_l = self.compute_loss(low_rank, sparse, features, labels) / lambda_low_rank
        U_s = self.compute_loss(low_rank, sparse, features, labels) / lambda_sparse
        n_tasks = low_rank.shape[1]

        low_ranks = []
        sparses = []

        for i in range(n_iter):

            # solve oracle for the low_rank part
            grad = self.compute_gradient(sparse, low_rank, features, labels)
            u, s, vh = svds(grad, k=1)
            D_l = - u @ vh
            if lambda_low_rank >= - np.trace(grad.T @ D_l):
                V_l = np.zeros(D_l.shape)
                Vt_l = 0
            else:
                V_l = U_l * D_l
                Vt_l = U_l

            # solve oracle for sparse part
            ind = np.unravel_index(np.argmax(np.abs(grad), axis=None), grad.shape)  # returns a tuple
            n, m = grad.shape
            e_i = np.zeros((n, 1))
            e_i[ind[0], 0] = 1
            e_jh = np.zeros((1, m))
            e_jh[0, ind[1]] = 1
            D_s = - np.sign(grad[ind[0], ind[1]]) * e_i @ e_jh

            if lambda_sparse >= - np.trace(grad.T @ D_s):
                V_s = np.zeros(D_s.shape)
                Vt_s = 0
            else:
                V_s = U_s * D_s
                Vt_s = U_s


            # linesearch

            # for j in range(n_tasks):
            #     b_l += np.linalg.norm(features[j] @ (low_rank[:, j] - V_l[:, j])) ** 2
            #     b_s += np.linalg.norm(features[j] @ (sparse[:, j] - V_s[:, j])) ** 2
            #     c += (V_s[:, j] - sparse[:, j]).T @ features[j].T @ features[j] @ (low_rank[:, j] - V_l[:, j])
            #     a_l += labels[:, j].T @ features[j] @ (low_rank[:, j] - V_l[:, j]) - (V_l[:, j] + V_s[:, j]).T \
            #            @ features[j].T @ features[j] @ (low_rank[:, j] - V_l[:, j])
            #     a_s += labels[:, j].T @ features[j] @ (sparse[:, j] - V_s[:, j]) - (V_s[:, j] + V_l[:, j]).T \
            #            @ features[j].T @ features[j] @ (sparse[:, j] - V_s[:, j])

            b_l = np.linalg.norm(np.einsum('ijk,ki->ji', features, low_rank - V_l, optimize=True)) ** 2
            b_s = np.linalg.norm(np.einsum('ijk,ki->ji', features, sparse - V_s, optimize=True)) ** 2
            c = np.einsum("ji,ikj,ikl,li", V_s -sparse, features, features, low_rank-V_l, optimize=True)
            a_l = np.einsum("ji,ijk,ki", labels, features, low_rank-V_l, optimize=True) - \
                  np.einsum("ji,ikj,ikl,li", V_l + V_s, features, features, low_rank-V_l, optimize=True)
            a_s = np.einsum("ji,ijk,ki", labels, features, sparse - V_s, optimize=True) - \
                  np.einsum("ji,ikj,ikl,li", V_l + V_s, features, features, sparse - V_s, optimize=True)


            a_l += lambda_low_rank * Vt_l - lambda_low_rank * t_l
            a_s += lambda_sparse * Vt_s - lambda_sparse * t_s

            gamma_l = (a_l * b_s + a_s * c) / (b_l * b_s - c ** 2)
            gamma_l = max(min(gamma_l, 1), 0)

            gamma_s = (a_s * b_l + a_l * c) / (b_l * b_s - c ** 2)
            gamma_s = max(min(gamma_s, 1), 0)

            # combine using gamma found with linesearch
            sparse = gamma_s * sparse + (1 - gamma_s) * V_s
            low_rank = gamma_l * low_rank + (1 - gamma_l) * V_l

            t_s = gamma_s * t_s + (1 - gamma_s) * Vt_s
            t_l = gamma_l * t_l + (1 - gamma_l) * Vt_l

            # additional proximal step
            sparse = self.soft_thresholding(sparse - lr * self.compute_gradient(sparse, low_rank, features, labels),
                                            lambda_sparse * lr)
            t_s = np.sum(np.abs(sparse))

            if i % 100 == 0:
                loss = self.compute_loss(sparse, low_rank, features, labels)
                loss += np.linalg.norm(low_rank, ord="nuc") * lambda_low_rank
                loss += np.sum(np.abs(sparse)) * lambda_sparse
                print(f"iteration {i}, loss={loss}")
                low_ranks.append(low_rank)
                sparses.append(sparse)

            # Update U_L and U_S again
            U_l = (self.compute_loss(sparse, low_rank, features,
                                     labels) + lambda_low_rank * t_l + lambda_sparse * t_s) / lambda_low_rank
            U_s = (self.compute_loss(sparse, low_rank, features,
                                     labels) + lambda_low_rank * t_l + lambda_sparse * t_s) / lambda_sparse

        return {"low_ranks": low_ranks, "sparses": sparses}

    def single_LSE(self, features, labels):
        """
        The method retrieves the parameters B = (Theta+Gamma) by fitting a least squares estimator on each task.
        :param features: The features matrices
        :param labels: The corresponding labels
        :return: The matrix B such that Y=XB
        """
        T = labels.shape[1]
        d = features[0].shape[1]
        B = np.zeros((d, T))
        for i in range(len(features)):
            B[:, i] = np.linalg.lstsq(features[i], labels[:, i], rcond=None)[0]
        return B
