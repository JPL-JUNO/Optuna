import logging
import sys

import optuna
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def objective(trial):
    iris = load_iris()
    classes = list(set(iris.target))
    train_X, valid_X, train_y, valid_y = train_test_split(iris.data, iris.target, test_size=.25, random_state=39)

    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    clf = SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_X, train_y, classes=classes)
        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_X, valid_y)
        trial.report(intermediate_value, step)
        # Report intermediate objective value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_X, valid_y)


optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
