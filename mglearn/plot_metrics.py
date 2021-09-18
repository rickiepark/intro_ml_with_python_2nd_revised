import numpy as np
import matplotlib.pyplot as plt

from .tools import plot_2d_separator, plot_2d_scores, cm, discrete_scatter
from .plot_helpers import ReBl


def plot_confusion_matrix_illustration():
    plt.figure(figsize=(8, 8))
    confusion = np.array([[402, 1], [6, 41]])
    plt.text(0.40, .7, confusion[0, 0], size=70, horizontalalignment='right')
    plt.text(0.40, .2, confusion[1, 0], size=70, horizontalalignment='right')
    plt.text(.90, .7, confusion[0, 1], size=70, horizontalalignment='right')
    plt.text(.90, 0.2, confusion[1, 1], size=70, horizontalalignment='right')
    plt.xticks([.25, .75], ["'9 아님' 예측", "'9' 예측"], size=20)
    plt.yticks([.25, .75], ["'9' 정답", "'9 아님' 정답"], size=20)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_binary_confusion_matrix():
    plt.text(0.45, .6, "TN", size=100, horizontalalignment='right')
    plt.text(0.45, .1, "FN", size=100, horizontalalignment='right')
    plt.text(.95, .6, "FP", size=100, horizontalalignment='right')
    plt.text(.95, 0.1, "TP", size=100, horizontalalignment='right')
    plt.xticks([.25, .75], ["음성 예측", "양성 예측"], size=15)
    plt.yticks([.25, .75], ["양성 클래스", "음성 클래스"], size=15)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_decision_threshold():
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X, y = make_blobs(n_samples=(400, 50), cluster_std=[7.0, 2],
                      random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    # plt.suptitle("decision_threshold")
    axes[0, 0].set_title("훈련 데이터")
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axes[0, 0])

    svc = SVC(gamma=.05).fit(X_train, y_train)
    axes[0, 1].set_title("임계값 0일 때")
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axes[0, 1])
    plot_2d_scores(svc, X_train, function="decision_function", alpha=.7,
                   ax=axes[0, 1], cm=ReBl)
    plot_2d_separator(svc, X_train, linewidth=3, ax=axes[0, 1])
    axes[0, 2].set_title("임계값 -0.8일 때")
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axes[0, 2])
    plot_2d_separator(svc, X_train, linewidth=3, ax=axes[0, 2], threshold=-.8)
    plot_2d_scores(svc, X_train, function="decision_function", alpha=.7,
                   ax=axes[0, 2], cm=ReBl)

    axes[1, 0].set_axis_off()

    mask = np.abs(X_train[:, 1] - 7) < 5
    bla = np.sum(mask)

    line = np.linspace(X_train.min(), X_train.max(), 100)
    axes[1, 1].set_title("임계값 0일 때 단면도")
    axes[1, 1].plot(line, svc.decision_function(np.c_[line, 10 * np.ones(100)]), c='k')
    dec = svc.decision_function(np.c_[line, 10 * np.ones(100)])
    contour = (dec > 0).reshape(1, -1).repeat(10, axis=0)
    axes[1, 1].contourf(line, np.linspace(-1.5, 1.5, 10), contour, alpha=0.4, cmap=cm)
    discrete_scatter(X_train[mask, 0], np.zeros(bla), y_train[mask], ax=axes[1, 1])
    axes[1, 1].set_xlim(X_train.min(), X_train.max())
    axes[1, 1].set_ylim(-1.5, 1.5)
    axes[1, 1].set_xticks(())
    axes[1, 1].set_ylabel("Decision value")

    contour2 = (dec > -.8).reshape(1, -1).repeat(10, axis=0)
    axes[1, 2].set_title("임계값 -0.8일 때 단면도")
    axes[1, 2].contourf(line, np.linspace(-1.5, 1.5, 10), contour2, alpha=0.4, cmap=cm)
    discrete_scatter(X_train[mask, 0], np.zeros(bla), y_train[mask], alpha=.1, ax=axes[1, 2])
    axes[1, 2].plot(line, svc.decision_function(np.c_[line, 10 * np.ones(100)]), c='k')
    axes[1, 2].set_xlim(X_train.min(), X_train.max())
    axes[1, 2].set_ylim(-1.5, 1.5)
    axes[1, 2].set_xticks(())
    axes[1, 2].set_ylabel("Decision value")
    axes[1, 0].legend(['negative class', 'positive class'])
