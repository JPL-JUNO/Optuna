import optuna
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import configparser
from time import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperParamsOpt:
    def __init__(self, data_X, data_y, model_list=None):
        if model_list is None:
            self.model_list = ['LogisticRegression', 'RandomForest', 'GBDT']
        self.X = data_X
        self.y = data_y
        self.model_list = model_list
        self.stop_time = 60
        self.trial_times = 10

    @staticmethod
    def _read_search_space():
        config_path = os.path.join(os.getcwd(), 'opt_config.ini')
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')

        return config

    def _objective(self, trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', self.model_list)
        config = self._read_search_space()

        tree_params = {
            'max_depth': trial.suggest_int('max_depth',
                                           config.getint('tree params', 'max_depth_min'),
                                           config.getint('tree params', 'max_depth_max')),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes',
                                                config.getint('tree params', 'max_leaf_nodes_min'),
                                                config.getint('tree params', 'max_leaf_nodes_min')),
            'min_samples_split': trial.suggest_int('min_samples_split',
                                                   config.getint('tree params', 'min_samples_split_min'),
                                                   config.getint('tree params', 'min_samples_split_max')),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                                  config.getint('tree params', 'min_samples_leaf_min'),
                                                  config.getint('tree params', 'min_samples_leaf_max')),
        }

        if classifier_name == 'GBDT':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', config.getint('estimators', 'n_estimators_min'),
                                                  config.getint('estimators', 'n_estimators_max')),
                'learning_rate': trial.suggest_float('learning_rate',
                                                     config.getfloat(classifier_name, 'learning_rate_min'),
                                                     config.getfloat(classifier_name, 'learning_rate_max')),
                'subsample': trial.suggest_float('subsample', config.getfloat(classifier_name, 'subsample_min'),
                                                 config.getfloat(classifier_name, 'subsample_max')),
                'max_features': trial.suggest_float('max_features',
                                                    config.getfloat(classifier_name, 'max_features_min'),
                                                    config.getfloat(classifier_name, 'max_features_max')),
            }
            classifier_obj = GradientBoostingClassifier(**params, **tree_params)
        elif classifier_name == 'XGBoost':
            params = {
                'lambda': trial.suggest_float('lambda', config.getfloat(classifier_name, 'lambda_min'),
                                              config.getfloat(classifier_name, 'lambda_max')),
                'alpha': trial.suggest_float('alpha', config.getfloat(classifier_name, 'alpha_min'),
                                             config.getfloat(classifier_name, 'alpha_max')),
                'subsample': trial.suggest_float('subsample', config.getfloat(classifier_name, 'subsample_min'),
                                                 config.getfloat(classifier_name, 'subsample_max')),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                        config.getfloat(classifier_name, 'colsample_bytree_min'),
                                                        config.getfloat(classifier_name, 'colsample_bytree_max'))
            }
            train_X, valid_X, train_y, valid_y = train_test_split(self.X, self.y, test_size=.25)
            data_train = xgb.DMatrix(train_X, label=train_y)
            data_valid = xgb.DMatrix(valid_X, label=valid_y)
            bst = xgb.train(params, data_train)
            pres = bst.predict(data_valid)
            pred_labels = np.rint(pres)
            accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
            return accuracy
        elif classifier_name == 'RandomForest':
            classifier_obj = RandomForestClassifier(**tree_params)
        else:
            params = {
                'C': trial.suggest_float('C', config.getfloat('LogisticRegression', 'lr_C_min'),
                                         config.getfloat('LogisticRegression', 'lr_C_max')),
            }
            classifier_obj = LogisticRegression(**params)

        score = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.y, n_jobs=-1, cv=5)
        accuracy = score.mean()

        return accuracy

    def opt_study(self) -> tuple:
        study = optuna.create_study(direction='maximize')
        start_time = time()
        study.optimize(self._objective, n_trials=self.trial_times, timeout=self.stop_time)
        end_time = time()
        opt_time = end_time - start_time
        if optuna.visualization.matplotlib.is_available():
            # optuna.visualization.matplotlib.plot_intermediate_values(study)
            # plt.savefig('intermediate_values.png', bbox_inches='tight')
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig('optimization_history.png', bbox_inches='tight')
        print('Optimization time: {:.2f} s'.format(opt_time))

        return (self._best_hyperparams(study),
                self._best_model(study),
                self._search_space_result(study),
                self._optimization_records(study))

    @staticmethod
    def _search_space_result(study) -> pd.DataFrame:
        """
        Returns the search space data frame
        :param study: optuna object
        :return: search_space_df
        """
        search_space_df = pd.DataFrame.from_dict(study.best_trial.distributions.items())
        search_space_df.columns = ['params', 'class']
        search_space_df.drop(labels=[0], axis=0, inplace=True)
        search_space_df['distribution'] = search_space_df['class'].apply(lambda x: str(type(x))[29: -14])
        search_space_df['high'] = search_space_df['class'].apply(lambda x: x.high)
        search_space_df['low'] = search_space_df['class'].apply(lambda x: x.low)
        search_space_df['step'] = search_space_df['class'].apply(lambda x: x.step)
        search_space_df['log'] = search_space_df['class'].apply(lambda x: x.log)
        search_space_df.drop(labels=['class'], axis=1, inplace=True)
        return search_space_df

    @staticmethod
    def _best_model(study) -> str:
        return study.best_params['classifier']

    @staticmethod
    def _best_hyperparams(study) -> pd.DataFrame:
        return pd.DataFrame(study.best_params, index=[0])

    @staticmethod
    def _optimization_records(study) -> pd.DataFrame:
        return study.trials_dataframe()


# test
if __name__ == '__main__':
    X, y = make_classification(n_samples=1000,
                               n_features=50,
                               n_informative=30,
                               n_redundant=10,
                               n_clusters_per_class=2,
                               random_state=39)
    opt = HyperParamsOpt(X, y, model_list=['XGBoost'])
    print(opt.opt_study())
