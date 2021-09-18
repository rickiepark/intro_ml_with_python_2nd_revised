import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import learning_curve, KFold

from .datasets import load_extended_boston


def plot_learning_curve(est, X, y):
    training_set_size, train_scores, test_scores = learning_curve(
        est, X, y, train_sizes=np.linspace(.1, 1, 20), cv=KFold(20, shuffle=True, random_state=1))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--',
                    label=estimator_name + " 훈련")
    plt.plot(training_set_size, test_scores.mean(axis=1), '-',
             label=estimator_name + " 테스트", c=line[0].get_color())
    plt.xlabel('훈련 세트 크기')
    plt.ylabel('점수 (R^2)')
    plt.ylim(0, 1.1)


def plot_ridge_n_samples():
    X, y = load_extended_boston()

    plot_learning_curve(Ridge(alpha=1), X, y)
    plot_learning_curve(LinearRegression(), X, y)
    plt.legend(loc=(0, 1.05), ncol=2, fontsize=11)
