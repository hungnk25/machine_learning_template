import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor

import time as tm

import logging
import logger_config
logger = logging.getLogger("main_logger")


#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost
EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric
EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric

class AutoLGB():
    
    default_lgb_space = {
        "n_estimators": scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        #"num_leaves": scope.int(hp.quniform("num_leaves", 2, LGBM_MAX_LEAVES, 1)),
        "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127, 255]),
        #"max_depth": scope.int(hp.quniform("max_depth", 2, LGBM_MAX_DEPTH, 1)),
        "max_depth": hp.choice("max_depth", [-1, 4, 6, 8, 10]),
        "min_child_samples": scope.int(hp.quniform("min_child_samples", 1, 256, 1)),            
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.1),
        "subsample": hp.quniform("subsample", 0.6, 1, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "verbosity": -1,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_error"
    }

    def __init__(self, space=default_lgb_space, n_eval=10):
        self.n_eval = n_eval
        self.space = space
        self.model = None
        self.best_params = None

    def tune(self, X, y):

        def objective(params):
            X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2)
            eval_set = [(X_val, y_val)]
            model = LGBMClassifier(**params)
            model.fit(X_trn, y_trn, eval_set=eval_set, verbose=False)
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            # TODO: Add the importance for the selected features
            #print("\tScore {0}".format(score))
            # The score function should return the loss (1-score)
            # since the optimize function looks for the minimum
            loss = 1 - score
            return {'loss': loss, 'status': STATUS_OK}
    
        logger.info(f"Starting optimization process for model LBM")
        st = tm.time()
        
        trials = Trials() 
        best = fmin(objective, self.space, algo=tpe.suggest, max_evals=self.n_eval, trials=trials)
        self.best_params = space_eval(self.space, best)
        logger.info(f"Best params for model: {self.best_params}")
        logger.info(f"Time taken to run for: {(tm.time() - st)/60:.1f}(mins)")

        #return self.best_params

    def fit(self, X, y):
        if not self.best_params:
            logger.info(f"Haven't run model tuning yet. Starting tuning to find best params")
            self.tune(X, y)
        logger.info(f"Fitting model with input data")
        model = LGBMClassifier(**self.best_params)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        logger.info(f"Model training accuracy: {acc}")
        self.model = model


    def predict(self, X):
        if not self.model:
            logger.info("There is no model. Tune and train your model first")
            return []
        y_pred = self.model.predict(X)
        return y_pred


class AutoXGB():
    
    default_xgb_space = {

        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
        'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
        'max_depth':  scope.int(hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'error',
        'objective': 'binary:logistic',
        'nthread': -1,
        'booster': 'gbtree',
        'tree_method': 'exact'
    }

    def __init__(self, space=default_xgb_space, n_eval=10):
        self.n_eval = n_eval
        self.space = space
        self.model = None
        self.best_params = None

    def tune(self, X, y):

        def objective(params):
            X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2)
            eval_set = [(X_val, y_val)]
            model = XGBClassifier(**params)
            model.fit(X_trn, y_trn, eval_set=eval_set, verbose=False)
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            # TODO: Add the importance for the selected features
            #print("\tScore {0}".format(score))
            # The score function should return the loss (1-score)
            # since the optimize function looks for the minimum
            loss = 1 - score
            return {'loss': loss, 'status': STATUS_OK}
    
        logger.info(f"Starting optimization process for model XgBoost Classifier")
        st = tm.time()
        
        trials = Trials() 
        best = fmin(objective, self.space, algo=tpe.suggest, max_evals=self.n_eval, trials=trials)
        self.best_params = space_eval(self.space, best)
        logger.info(f"Best params for model: {self.best_params}")
        logger.info(f"Time taken to run for: {(tm.time() - st)/60:.1f}(mins)")

        #return self.best_params

    def fit(self, X, y):
        if not self.best_params:
            logger.info(f"Haven't run model tuning yet. Starting tuning to find best params")
            self.tune(X, y)
        logger.info(f"Fitting model with input data")
        model = XGBClassifier(**self.best_params)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        logger.info(f"Model training accuracy: {acc}")
        self.model = model


    def predict(self, X):
        if not self.model:
            logger.info("There is no model. Tune and train your model first")
            return []
        y_pred = self.model.predict(X)
        return y_pred


#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric


class AutoCatboost():
    
    default_catboost_space = {

        "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
        'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
        'max_bin' : scope.int(hp.quniform('max_bin', 1, 254, 1)), #if using CPU just set this to 254
        'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
        'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
        'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
        'bootstrap_type' : hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'No']),
        'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.25)),
        'eval_metric' : "Accuracy",
        'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
        'od_type' : 'Iter',
        'od_wait' : 25,
        #'task_type' : 'GPU',
        'verbose' : False          
    }

    def __init__(self, space=default_catboost_space, n_eval=10):
        self.n_eval = n_eval
        self.space = space
        self.model = None
        self.best_params = None

    def tune(self, X, y):

        def objective(params):
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            model = CatBoostClassifier(**params)
            scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy", n_jobs=-1)
            #y_pred = model.predict(x_val)
            score = scores.mean()
            # TODO: Add the importance for the selected features
            #print("\tScore {0}".format(score))
            # The score function should return the loss (1-score)
            # since the optimize function looks for the minimum
            loss = 1 - score
            return {'loss': loss, 'status': STATUS_OK}
    
        logger.info(f"Starting optimization process for model Catboost Classifier")
        st = tm.time()
        
        trials = Trials() 
        best = fmin(objective, self.space, algo=tpe.suggest, max_evals=self.n_eval, trials=trials)
        self.best_params = space_eval(self.space, best)
        logger.info(f"Best params for model: {self.best_params}")
        logger.info(f"Time taken to run for: {(tm.time() - st)/60:.1f}(mins)")

        #return self.best_params

    def fit(self, X, y):
        if not self.best_params:
            logger.info(f"Haven't run model tuning yet. Starting tuning to find best params")
            self.tune(X, y)
        logger.info(f"Fitting model with input data")
        model = CatBoostClassifier(**self.best_params)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        logger.info(f"Model training accuracy: {acc}")
        self.model = model


    def predict(self, X):
        if not self.model:
            logger.info("There is no model. Tune and train your model first")
            return []
        y_pred = self.model.predict(X)
        return y_pred