import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm
from forecasting.Models.local_linear_forest import LocalLinearForestRegressor
from forecasting.data_preprocessing.obtain_data import obtainData

def objective(trial, model_name, X_train, Y_train, X_ridge=None):
    if model_name == "RandomForest":
        parameters = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100,200, 300, 400, 500]),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5)
        }
        model = RandomForestRegressor(**parameters)
    elif model_name == "LocalLinearForest":
        parameters = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100,200, 300, 400, 500]),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5),
            'lam' : trial.suggest_categorical('lam', [0.01, 0.1,0.25, 0.5, 0.75, 1,])
        }
        model = LocalLinearForestRegressor(**parameters)
        if X_ridge is not None:
            model._X_train_ridge = X_ridge

    # Perform cross-validation
    mse_scores = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    mean_mse = mse_scores.mean()
    return mean_mse

def hypertune_model(model_name, X_train, Y_train, X_ridge=None, n_trials = 100):
    print()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner(), sampler=TPESampler(seed = 42))
    study.optimize(lambda trial: objective(trial, model_name, X_train, Y_train, X_ridge), n_trials=n_trials, show_progress_bar=True, n_jobs =-1)
    best_params = study.best_params
    print(model_name, best_params)
    return best_params

def run_hypertune(n_trials):
    coins = ["Bitcoin", "Ethereum", "Tether", "Binance Coin", "Bitcoin Cash", "Litecoin", "Internet Computer", "Polygon"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results_rf = pd.DataFrame(columns=['Coin'])
    results_llf = pd.DataFrame(columns=['Coin'])

    for coin in tqdm(coins):
        X, Y, X_ridge = obtainData(coin)  # Assuming obtainData returns X, Y, X_ridge
        initial_train_size = 0.7
        train_size = int(len(X) * initial_train_size)
        X_train_split = X.iloc[:train_size].to_numpy()
        X_train_ridge = X_ridge.iloc[:train_size].to_numpy()
        Y_train = Y.iloc[:train_size].to_numpy().ravel()

        # Hyperparameter tuning for Random Forest
        best_rf_params = hypertune_model("RandomForest", X_train_split, Y_train, n_trials= n_trials)
        best_rf_params['Coin'] = coin
        print("Best RF Params:", best_rf_params)
        results_rf = pd.concat([results_rf, pd.DataFrame([best_rf_params])])

        # Hyperparameter tuning for Local Linear Forest
        best_llf_params = hypertune_model("LocalLinearForest", X_train_split, Y_train, X_train_ridge, n_trials = n_trials)
        best_llf_params['Coin'] = coin
        print("Best LLF Params:", best_llf_params)
        results_llf = pd.concat([results_llf, pd.DataFrame([best_llf_params])])

    # Convert DataFrame to LATEX tables
    latex_table_rf = results_rf.to_latex(index=False, float_format="%.3f")
    latex_table_llf = results_llf.to_latex(index=False, float_format="%.3f")

    print(latex_table_rf)
    print(latex_table_llf)