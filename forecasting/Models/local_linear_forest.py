import numpy as np
from sklearn.ensemble import RandomForestRegressor

class LocalLinearForestRegressor(RandomForestRegressor):
    def __init__(self,
                 n_estimators=500,
                 criterion='squared_error',
                 max_depth=10,
                 min_samples_split=3,
                 min_samples_leaf=2,
                 min_weight_fraction_leaf=0.,
                 max_features=0.7,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.01,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=True,
                 lam = 1,
                 max_samples = 0.5):
        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         max_samples = max_samples)

        self._incidence_matrix = None
        self._X_train_split = None
        self._X_train_ridge = None
        self._Y_train = None
        self.lam = lam

    def _extract_leaf_nodes_ids(self, X):
        leafs = [e.apply(X).reshape(-1, 1) for e in self.estimators_]
        leaf_nodes_ids = np.concatenate(leafs, axis=1)
        assert leaf_nodes_ids.shape[0] == X.shape[0]
        assert leaf_nodes_ids.shape[1] == len(self.estimators_)
        return leaf_nodes_ids

    def fit(self, X_split, y, X_ridge = None, sample_weight=None):
        super().fit(X_split, y, sample_weight=sample_weight)
        self._X_train_split = X_split
        self._X_train_ridge = X_ridge
        self._Y_train = y
        self._incidence_matrix = self._extract_leaf_nodes_ids(X_split)
        return self

    def _get_forest_coefficients(self, observation_leaf_ids):
        coeffs = np.zeros(self._X_train_split.shape[0])
        for j in range(observation_leaf_ids.shape[1]):
            matching_nodes = (self._incidence_matrix[:, j] == observation_leaf_ids[0, j])
            counts = np.sum(matching_nodes)
            if counts > 0:
                coeffs += matching_nodes / counts
        return coeffs / self.n_estimators

    def predict_LLF(self, X_test_split, X_test_ridge):
        results = []
        X_test_split = np.array(X_test_split)
        X_test_ridge = np.array(X_test_ridge)

        for i in range(X_test_split.shape[0]):
            x0_split = X_test_split[i, :].reshape(1, -1)
            x0_ridge = X_test_ridge[i, :].reshape(1, -1)
            actual_leaf_ids = self._extract_leaf_nodes_ids(x0_split)
            alpha_i = self._get_forest_coefficients(actual_leaf_ids)

            Delta = np.hstack([np.ones((self._X_train_ridge.shape[0], 1)), self._X_train_ridge - x0_ridge])
            A = np.diag(alpha_i)
            d = self._X_train_ridge.shape[1]
            J = np.diag([0] + [1] * d)

            inv_term = np.linalg.inv(Delta.T @ A @ Delta + self.lam * J)
            theta_hat = inv_term @ Delta.T @ A @ self._Y_train

            u_hat = theta_hat[0]
            results.append(u_hat)

        return np.array(results).reshape(-1)