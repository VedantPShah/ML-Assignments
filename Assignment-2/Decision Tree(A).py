import numpy as np
import os
import graphviz


def partition(x): 
   
   d={}
   
   for v in np.unique(x):
     d.update({v: (x == v).nonzero()[0]})
   
   return d
   
   raise Exception('Function not yet implemented!')

def entropy(y):
    
    values,counts=np.unique(y,return_counts=True)
    e=0
    
    for i in range(len(values)):
        p=counts[i]/len(y)
        e=e-p*np.log2(p)
    
    return e
    
    raise Exception('Function not yet implemented!')
    
def mutual_information(x, y):
    
    Ey=entropy(y)
    values,counts=np.unique(x,return_counts=True)
    
    for v in values:
      Ey=Ey-((len(y[x==v])/len(y))*entropy(y[x==v])+(len(y[x!=v])/len(y))*entropy(y[x!=v]))
      
    return Ey
    
    raise Exception('Function not yet implemented!')
    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    
    tree = {}
    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, v) for v in np.unique(x[:, i])] for i in range(x.shape[1])])
    y_values, y_counts = np.unique(y, return_counts=True)

    if len(y_values) == 1:
        return y_values[0]
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return y_values[np.argmax(y_counts)]
    mutual_info = np.array([mutual_information(np.array(x[:, i] == v).astype(int), y)
                                 for (i, v) in attribute_value_pairs])
    (attr, value) = attribute_value_pairs[np.argmax(mutual_info)]
    partitions = partition(np.array(x[:, attr] == value).astype(int))
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(np.all(attribute_value_pairs == (attr, value), axis=1)), 0)

    for split_value, indices in partitions.items():
        x_subset = x.take(indices, axis=0)
        y_subset = y.take(indices, axis=0)
        decision = bool(split_value)

        tree[(attr, value, decision)] = id3(x_subset, y_subset, attribute_value_pairs=attribute_value_pairs,
                                            depth=depth + 1, max_depth=max_depth) 
    
    return tree

    raise Exception('Function not yet implemented!')
    
def predict_example(x, tree):
    if type(tree) is not dict:
        return tree
    else:
        for s, val in tree.items():
            s_index = s[0]
            s_value = s[1]
            s_decision = s[2]

            if s_decision == (x[s_index] == s_value):
                if type(val) is dict:
                    label = predict_example(x, val)
                else:
                    label = val

                return label
        
    raise Exception('Function not yet implemented!')
    
def compute_error(y_true, y_pred):
   
    n = len(y_true)
    err = [y_true[i] != y_pred[i] for i in range(n)]
    return sum(err) / n
    
    raise Exception('Function not yet implemented!')

def pretty_print(tree, depth=0):
   
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))
            
def render_dot_file(dot_string, save_file, image_format='png'):
    
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)
    
def to_graphviz(tree, dot_string='', uid=-1, depth=0):

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('C:/Users/Vedant/Downloads/monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('C:/Users/Vedant/Downloads/monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    y_pred_train = [predict_example(x, decision_tree) for x in Xtrn]
    train_error = compute_error(ytrn, y_pred_train)
    print('Training error={0:4.2f}%'.format(train_error*100))
            
