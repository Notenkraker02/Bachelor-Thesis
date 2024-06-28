import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from simulation_study.models.local_linear_forest import LocalLinearForestRegressor
import xgboost as xgb
from bartpy.sklearnmodel import SklearnModel
from simulation_study.models.lasso_rf import LassoRandomForest

def objective(trial, model_name, X_train, Y_train, X_ridge=None):
    if model_name == "RandomForest":
        parameters = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
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
            'lam' : trial.suggest_categorical('lam', [0.01, 0.05, 0.1, 0.2])
        }
        model = LocalLinearForestRegressor(**parameters)
        if X_ridge is not None:
            model._X_train_ridge = X_ridge

    elif model_name == "XGBoost":
         parameters = { 
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.5, 1.0)
         }
         model = xgb.XGBRegressor(**parameters)

    elif model_name == "BART":
        parameters = {
            'n_chains': trial.suggest_int('n_chains', 2, 4),
            'n_trees': trial.suggest_categorical('n_trees', [50, 100, 200]),
            'n_burn': trial.suggest_categorical('n_burn', [100, 200, 300]),
            'n_samples': trial.suggest_categorical('n_samples', [500, 1000]),
        }
        model = SklearnModel(**parameters)

    elif model_name == "LASSO-RF":
        parameters = {
            'lasso_alpha': trial.suggest_float('lasso_alpha', 0.001, 0.1),
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5)
        }
        model = LassoRandomForest(**parameters)

    # Perform cross-validation
    mse_scores = -cross_val_score(model, X_train, Y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    mean_mse = mse_scores.mean()
    return mean_mse

def hypertune_model(model_name, X_train, Y_train, X_ridge=None, n_trials=100):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner(), sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, model_name, X_train, Y_train, X_ridge), n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
    best_params = study.best_params
    print(model_name, best_params)
    return best_params

