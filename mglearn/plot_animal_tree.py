from imageio import imread
import matplotlib.pyplot as plt


def plot_animal_tree(ax=None):
    import graphviz
    plt.figure(dpi=100)
    if ax is None:
        ax = plt.gca()
    mygraph = graphviz.Digraph(node_attr={'shape': 'box'},
                               edge_attr={'labeldistance': "10.5"},
                               format="png")
    mygraph.node("0", "날개가 있나요?")
    mygraph.node("1", "날 수 있나요?")
    mygraph.node("2", "지느러미가 있나요?")
    mygraph.node("3", "매")
    mygraph.node("4", "펭귄")
    mygraph.node("5", "돌고래")
    mygraph.node("6", "곰")
    mygraph.edge("0", "1", label="True")
    mygraph.edge("0", "2", label="False")
    mygraph.edge("1", "3", label="True")
    mygraph.edge("1", "4", label="False")
    mygraph.edge("2", "5", label="True")
    mygraph.edge("2", "6", label="False")
    mygraph.render("tmp")
    ax.imshow(imread("tmp.png"))
    ax.set_axis_off()
