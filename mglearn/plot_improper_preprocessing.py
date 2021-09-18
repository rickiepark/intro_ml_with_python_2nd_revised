import matplotlib.pyplot as plt


def make_bracket(s, xy, textxy, width, ax):
    annotation = ax.annotate(
        s, xy, textxy, ha="center", va="center", size=20,
        arrowprops=dict(arrowstyle="-[", fc="w", ec="k",
                        lw=2,), bbox=dict(boxstyle="square", fc="w"))
    annotation.arrow_patch.get_arrowstyle().widthB = width


def plot_improper_processing():
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15],
                         color=['white', 'grey', 'grey'], hatch="//",
                         align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-0.5, 4.5)
        axis.set_xlim(0, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.8, "훈련 폴드",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(13.5, -.8, "검증 폴드",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(17.5, -.8, "테스트 세트",
                  fontdict={'fontsize': 14}, horizontalalignment="center")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[0])
    make_bracket("SVC fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket("SVC predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("교차 검증")
    axes[1].set_title("테스트 세트 예측")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket("SVC fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket("SVC predict", (17.5, 3), (17.5, 4), 4.8, axes[1])
    fig.subplots_adjust(hspace=.3)


def plot_proper_processing():
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9],
                         left=[0, 12, 15], color=['white', 'grey', 'grey'], hatch="//",
                         align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-.5, 4.5)
        axis.set_xlim(0, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.8, "훈련 폴드", fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(13.5, -.8, "검증 폴드", fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(17.5, -.8, "테스트 세트", fontdict={'fontsize': 14}, horizontalalignment="center")

    make_bracket("scaler fit", (6, 1.3), (6, 2.), 12, axes[0])
    make_bracket("SVC fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket("SVC predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("교차 검증")
    axes[1].set_title("테스트 세트 예측")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket("SVC fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket("SVC predict", (17.5, 3), (17.5, 4), 4.8, axes[1])
    fig.subplots_adjust(hspace=.3)
