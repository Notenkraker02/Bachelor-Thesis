import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class LassoRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, lasso_alpha=1.0, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.lasso_alpha = lasso_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        n = len(y)
        self.inds = np.random.choice(range(n), size=n//2, replace=False)
        self.inds_not = np.array([i for i in range(n) if i not in self.inds])

        # Train Lasso on the first half of the data
        self.lasso = LassoCV(alphas=[self.lasso_alpha], cv=5, max_iter=50, tol=1e-2)
        self.lasso.fit(X[self.inds], y[self.inds])
        lasso_preds = self.lasso.predict(X[self.inds_not])
        self.lasso_resids = y[self.inds_not] - lasso_preds

        # Train Random Forest on the second half of the data using the residuals from Lasso
        self.forest = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, 
                                            random_state=self.random_state)
        self.forest.fit(X[self.inds_not], self.lasso_resids)

    def predict(self, X):
        rf_preds = self.forest.predict(X)
        lasso_preds_test = self.lasso.predict(X)
        return rf_preds + lasso_preds_test
