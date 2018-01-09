import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', facecolor='0.8')
leaf_node = dict(boxstyle='round4', facecolor='0.8')

def plot_node(ax, text, center_pt, parent_pt, node_type):
    ax.annotate(text, xy=parent_pt, xycoords='axes fraction', 
        xytext=center_pt, textcoords='axes fraction',
        verticalalignment='center', horizontalalignment='center',
        bbox=node_type, arrowprops=dict(facecolor='black', arrowstyle='<-'))

def plot_midtext(ax, center_pt, parent_pt, text):
    xmid = (parent_pt[0] - center_pt[0]) / 2 + center_pt[0]
    ymid = (parent_pt[1] - center_pt[1]) / 2 + center_pt[1]
    ax.text(xmid, ymid, text, verticalalignment='center', horizontalalignment='center', rotation=30)

def create_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_node(ax, 'decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(ax, 'leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plot_midtext(ax, (0.5, 0.1), (0.1, 0.5), 'decison')
    plot_midtext(ax, (0.8, 0.1), (0.3, 0.8), 'leaf')
    fig.show()

def plot_tree(tree, parent_pt):
