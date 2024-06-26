from sklearn.ensemble import RandomForestRegressor

def train_test_rf(X_train, Y_train, X_test, n_estimators = 500, max_depth = 10, max_features = 0.7, min_samples_split = 4, min_samples_leaf = 2):
    forest = RandomForestRegressor(n_estimators= n_estimators, max_depth= max_depth, max_features = max_features, 
                                   min_samples_split = min_samples_split, min_samples_leaf= min_samples_leaf, n_jobs= -1)
    forest.fit(X_train, Y_train)
    predictions = forest.predict(X_test)

    return predictions
