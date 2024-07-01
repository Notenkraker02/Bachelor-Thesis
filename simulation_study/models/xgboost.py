import xgboost as xgb

def predict_xgboost(X_train, Y_train, X_test, n_estimators=500, max_depth=10, learning_rate=0.5):
    xg_boost = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    xg_boost.fit(X_train, Y_train, verbose=False)

    xg_preds = xg_boost.predict(X_test)
    return xg_preds
