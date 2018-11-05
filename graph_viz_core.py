import pandas as pd
import random
import numpy as np
import os
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from matplotlib import pylab


def read_data(path):
    db = pd.read_csv(path, sep=";")
    # delete time
    db = db.drop(['PurchaseDate'], axis=1)
    # oneHot cities and group name
    # 1260 городов - delete:)
    db = db.drop(['City'], axis=1)
    one_hot_2 = pd.get_dummies(db['GroupTNName'])
    db = db.join(one_hot_2)
    db = db.drop(['GroupTNName'], axis=1)
    X = db.values
    # from ['12640,00' '5023,00' '7815,00' ... '655,00' '10466,00' '341,10'] to [12640.0 5023.0 7815.0 ... 655.0 10466.0 341.1]
    X[:, 1] = [float(x[:x.find(',')] + '.' + x[x.find(',') + 1:]) for x in X[:, 1]]
    y = [random.choice([1, 0]) for _ in range(X.shape[0])]
    return X, np.asarray(y)


def visual(dtree):
    dot_data = "my_tree.dot"
    export_graphviz(dtree, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True)
    os.system("dot -Tpng " + dot_data+ " -o my_tree.png")


def visual_path(nodes, feature, features_names, threshold, value, class_names, sides, sample_path=False, X_test=None, sample_id=None):
    prev = None
    prev_node_id = 0
    node_num = 1
    first_string = ['digraph G {',
                    '{',
                    'node [style="rounded, filled" fontcolor=black fontsize=10 shape=underline];']
    second_string = []

    print_string = ['If']

    for node_id in nodes:
        side = sides[node_id]
        if nodes[-1] == node_id:
            print_string.append('then')
            if not side:
                left = node_num
                first_string.append('%s [color=coral1 label=""];' % node_num)
                node_num += 1

                right = node_num
                # vs = " ".join(str(v) for v in value[node_id])
                # label = f"{vs}\nclass = {np.argmax(value[node_id])}"
                label1 = f'{class_names[int(np.argmax(value[node_id]))]}'  + '<br align="left"/>'
                label2 = ''
                for i, clas in enumerate(class_names):
                    v = value[node_id]
                    label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                first_string.append(
                    f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
                node_num += 1

                print_string.append(label)

                second_string.append(f'{prev} -> {left} [arrowhead=onormal, color=coral1];')
                label = ""
                if sample_path:
                    label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=palegreen3, label="{label}"];')
            elif side:
                left = node_num
                # vs = " ".join(str(v) for v in value[node_id])
                # label = f"{vs}\nclass = {np.argmax(value[node_id])}"
                label1 = f'{class_names[int(np.argmax(value[node_id]))]}' + '<br align="left"/>'

                label2 = ''
                for i, clas in enumerate(class_names):
                    v = value[node_id]
                    label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                first_string.append(f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
                node_num += 1

                right = node_num
                first_string.append(f'{node_num} [color=coral1  label=""];')
                node_num += 1

                print_string.append(label)

                second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=coral1];')
                label = ""
                if sample_path:
                    label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                    label = label + ' ' * (len(label) + 1)
                second_string.append(
                    f'{prev} -> {left} [arrowhead=onormal, color=palegreen3, headlabel="{label}"  labeldistance=2.5 labelangle=30];')


            continue

        if not prev:
            label1 = "%s &lt;= %s" % (features_names[feature[node_id]], round(threshold[node_id], 2)) + '<br align="left"/>'

            print_string.append(label1)

            label2 = ''
            for i, clas in enumerate(class_names):
                v = value[node_id]
                label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

            label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'
            first_string.append(
                f'{node_num} [style=filled color=white fillcolor=palegreen3 label={label}];')
            prev = node_num
            prev_node_id = node_id
            node_num += 1
        elif prev and not side:
            left = node_num
            first_string.append(f'{node_num} [color=coral1 label=""];')
            node_num += 1

            right = node_num

            label1 = "%s &lt;= %s" % (features_names[feature[node_id]], round(threshold[node_id], 2))  + '<br align="left"/>'

            print_string.append('and')
            print_string.append(label1)

            label2 = ''
            for i, clas in enumerate(class_names):
                v = value[node_id]
                label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

            label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

            first_string.append(
                f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
            node_num += 1

            second_string.append(f'{prev} -> {left} [arrowhead=onormal, color=coral1];')
            label = ""
            if sample_path:
                label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
            second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=palegreen3, label="{label}"];')
            prev = right
            prev_node_id = node_id
        elif prev and side:
            left = node_num
            label1 = "%s &lt;= %s" % (features_names[feature[node_id]], round(threshold[node_id], 2)) + '<br align="left"/>'

            print_string.append('and')
            print_string.append(label1)

            label2 = ''
            for i, clas in enumerate(class_names):
                v = value[node_id]
                label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

            label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

            first_string.append(f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3  label={label}]')
            node_num += 1

            right = node_num
            first_string.append(f'{node_num} [color=coral1 label=""];')
            node_num += 1

            second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=coral1];')
            label = ""
            if sample_path:
                label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                label = label + ' ' * len(label)
            second_string.append(
                f'{prev} -> {left} [arrowhead=onormal, color=palegreen3, headlabel="{label}" labeldistance=2.5 labelangle=30];')

            prev = left
            prev_node_id = node_id


    first_string.append('}\n')
    second_string.append('}')
    # print("\n".join(s for s in first_string))
    # print("\n".join(s for s in second_string))
    with open('write.dot', 'w') as f:
        f.write("\n".join(s for s in first_string))
        f.write("\n".join(s for s in second_string))

    import os
    os.system('dot write.dot -Tpng -o write.png')
    # from graphviz import Source
    # G = Source.from_file('write.dot')
    # G.render()
    # print(' '.join(s for s in print_string))

def find_path(model, X_test):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    features_names = ['Price', 'Quantity', 'Sale', 'Quality']
    threshold = model.tree_.threshold
    value = model.tree_.value
    parent_id = [-1 for _ in range(len(value))]
    class_names = ['Sold', 'Not Sold', 'Maybe Sold']

    zs = []

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    sides=[False for _ in range(len(value))]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            parent_id[children_left[node_id]] = node_id
            sides[children_left[node_id]] = True

            stack.append((children_right[node_id], parent_depth + 1))
            parent_id[children_right[node_id]] = node_id
            sides[children_right[node_id]] = False
        else:
            is_leaves[node_id] = True
            if np.max(value[node_id]) >= 30.:
                zs.append(node_id)

    for z in zs[:2]:
        node = z
        path = [node]
        while parent_id[node] != -1:
            path.append(parent_id[node])
            node = parent_id[node]
        # print(z, value[z], path[::-1])
        visual_path(path[::-1], feature, features_names, threshold, value, class_names, sides)

    node_indicator = model.decision_path(X_test)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    # visual_path(node_index, feature, features_names, threshold, value, class_names, sides, True, X_test, sample_id)

def run_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    visual(tree)
    find_path(tree, X_test)
    print("Accuracy of Decision Tree is", accuracy_score(y_test, y_pred) * 100)


def main():
    #    X, y = read_data("purchase_test.csv")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target  # три класса
    run_models(X, y)


# start = time.time()
# main()
# print("Time to do it all =", (time.time() - start) / 60, "minutes")

import base64
from io import BytesIO
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

_X = np.random.randint(0, 100, size=(70000, 10))
_y = np.array([int(row.sum() > 500) for row in _X])
MODEL = DecisionTreeClassifier(min_samples_leaf=100).fit(_X, _y)


def viz(json_data, target, groups):
    predictions= []
    X = pd.read_json(json_data)
    y = X[target].mean()
    X.drop(target, axis=1, inplace=True)

    n_nodes = MODEL.tree_.node_count
    children_left = MODEL.tree_.children_left
    children_right = MODEL.tree_.children_right
    feature = MODEL.tree_.feature
    features_names = [chr(97 + i) for i in range(len(feature))]
    threshold = MODEL.tree_.threshold
    value = MODEL.tree_.value
    parent_id = [-1 for _ in range(len(value))]
    class_names = ['Sold', 'Not Sold']

    zs = []

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    sides = [False for _ in range(len(value))]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            parent_id[children_left[node_id]] = node_id
            sides[children_left[node_id]] = True

            stack.append((children_right[node_id], parent_depth + 1))
            parent_id[children_right[node_id]] = node_id
            sides[children_right[node_id]] = False
        else:
            is_leaves[node_id] = True
            if np.max(value[node_id]) >= 30.:
                zs.append(node_id)

    for z in zs[:2]:
        node = z
        path = [node]
        while parent_id[node] != -1:
            path.append(parent_id[node])
            node = parent_id[node]
        # print(z, value[z], path[::-1])

        nodes = path[::-1]
        sample_path = False
        prev = None
        prev_node_id = 0
        node_num = 1
        first_string = ['digraph G {',
                        '{',
                        'node [style="rounded, filled" fontcolor=black fontsize=10 shape=underline];']
        second_string = []

        print_string = ['If']

        for node_id in nodes:
            side = sides[node_id]
            if nodes[-1] == node_id:
                print_string.append('then')
                if not side:
                    left = node_num

                    first_string.append(f'{node_num} [color=coral1 label=""]')
                    node_num += 1

                    right = node_num
                    # vs = " ".join(str(v) for v in value[node_id])
                    # label = f"{vs}\nclass = {np.argmax(value[node_id])}"
                    label1 = f'{class_names[int(np.argmax(value[node_id]))]}' + '<br align="left"/>'
                    label2 = ''
                    for i, clas in enumerate(class_names):
                        v = value[node_id]
                        label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                    label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                    first_string.append(
                        f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
                    node_num += 1

                    print_string.append(label)

                    second_string.append(f'{prev} -> {left} [arrowhead=onormal, color=coral1];')
                    label = ""
                    if sample_path:
                        label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                    second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=palegreen3, label="{label}"];')
                elif side:
                    left = node_num
                    # vs = " ".join(str(v) for v in value[node_id])
                    # label = f"{vs}\nclass = {np.argmax(value[node_id])}"
                    label1 = f'{class_names[int(np.argmax(value[node_id]))]}' + '<br align="left"/>'

                    label2 = ''
                    for i, clas in enumerate(class_names):
                        v = value[node_id]
                        label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                    label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                    first_string.append(
                        f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
                    node_num += 1

                    right = node_num
                    first_string.append(f'{node_num} [color=coral1  label=""];')
                    node_num += 1

                    print_string.append(label)

                    second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=coral1];')
                    label = ""
                    if sample_path:
                        label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                        label = label + ' ' * (len(label) + 1)
                    second_string.append(
                        f'{prev} -> {left} [arrowhead=onormal, color=palegreen3, headlabel="{label}"  labeldistance=2.5 labelangle=30];')

                continue

            if not prev:
                label1 = "%s &lt;= %s" % (
                features_names[feature[node_id]], round(threshold[node_id], 2)) + '<br align="left"/>'

                print_string.append(label1)

                label2 = ''
                for i, clas in enumerate(class_names):
                    v = value[node_id]
                    label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'
                first_string.append(
                    f'{node_num} [style=filled color=white fillcolor=palegreen3 label={label}];')
                prev = node_num
                prev_node_id = node_id
                node_num += 1
            elif prev and not side:
                left = node_num
                first_string.append(f'{node_num} [color=coral1 label=""];')
                node_num += 1

                right = node_num

                label1 = "%s &lt;= %s" % (
                features_names[feature[node_id]], round(threshold[node_id], 2)) + '<br align="left"/>'

                print_string.append('and')
                print_string.append(label1)

                label2 = ''
                for i, clas in enumerate(class_names):
                    v = value[node_id]
                    label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                first_string.append(
                    f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3 label={label}];')
                node_num += 1

                second_string.append(f'{prev} -> {left} [arrowhead=onormal, color=coral1];')
                label = ""
                if sample_path:
                    label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=palegreen3, label="{label}"];')
                prev = right
                prev_node_id = node_id
            elif prev and side:
                left = node_num
                label1 = "%s &lt;= %s" % (
                features_names[feature[node_id]], round(threshold[node_id], 2)) + '<br align="left"/>'

                print_string.append('and')
                print_string.append(label1)

                label2 = ''
                for i, clas in enumerate(class_names):
                    v = value[node_id]
                    label2 += f'{clas}: {v[0][i]} ({round(v[0][i] * 100 / np.sum(v), 1)}%)' + '<br align="left" />'

                label = f'<<FONT POINT-SIZE="10" ><u>{label1}</u></FONT><FONT POINT-SIZE="9">{label2}</FONT>>'

                first_string.append(f'{node_num} [style=filled color=palegreen3 fillcolor=palegreen3  label={label}]')
                node_num += 1

                right = node_num
                first_string.append(f'{node_num} [color=coral1 label=""];')
                node_num += 1

                second_string.append(f'{prev} -> {right} [arrowhead=onormal, color=coral1];')
                label = ""
                if sample_path:
                    label = f'{features_names[feature[prev_node_id]]} = {X_test[sample_id, feature[prev_node_id]]}'
                    label = label + ' ' * len(label)
                second_string.append(
                    f'{prev} -> {left} [arrowhead=onormal, color=palegreen3, headlabel="{label}" labeldistance=2.5 labelangle=30];')

                prev = left
                prev_node_id = node_id

        first_string.append('}\n')
        second_string.append('}')
        # print("\n".join(s for s in first_string))
        # print("\n".join(s for s in second_string))
        with open('write.dot', 'w') as f:
            f.write("\n".join(s for s in first_string))
            f.write("\n".join(s for s in second_string))

        # import os
        from graphviz import Source
        # G = Source.from_file('write.dot')
        # G.render(filename='img.png', format='png')
        # G.draw('img.png')
        # Source('write.dot', img, format='png')
        # os.system(f'dot write.dot -Tpng -o {img}.png')
        os.system('dot -Tpng write.dot -o img.png')
        with open("img.png", "rb") as imageFile:
            img_in_bytes = base64.b64encode(imageFile.read())

        # filename = G.render(filename=img)
        # pylab.savefig(img, format='png')
        predictions.append({
            'plot': str(img_in_bytes)
        })
        print(predictions)
    return predictions