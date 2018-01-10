import matplotlib.pyplot as plt
from decision_tree import get_tree_depth, get_leafs_count

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

def plot_tree(ax, tree, parent_pt, delta_height, delta_width):
    label = list(tree.keys())[0]
    child_count = len(tree[label])
    xoffset = parent_pt[0] - (child_count // 2 - 0.5) * delta_width
    ynode = parent_pt[1] - delta_width
    plot_node(ax, label, parent_pt, parent_pt, decision_node)  # plot the center node    
    for index, (val, subtree) in enumerate(tree[label].items()):
        node_pt =  [xoffset + index * delta_width, ynode]
        plot_midtext(ax, node_pt, parent_pt, val)                    
        if isinstance(subtree, dict):
            plot_node(ax, '', node_pt, parent_pt, decision_node)            
            plot_tree(ax, subtree, node_pt, delta_height, delta_width)
        else:
            plot_node(ax, subtree, node_pt, parent_pt, leaf_node)

def create_tree_plot(tree):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    depth = get_tree_depth(tree)
    leafs = get_leafs_count(tree)
    delta_height = 1 / (depth - 1)
    delta_width = 1 / leafs
    plot_tree(ax, tree, (0.5, 1), delta_height, delta_width)
