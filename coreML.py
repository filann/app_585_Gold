from io import BytesIO

import numpy as np
import pandas as pd
import base64
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from collections import Counter
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from itertools import chain, combinations
import os
import random
import datetime
from jinja2 import Template


if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('trees'):
    os.makedirs('trees')

# Функция preprocess_data обрабатывает входные данные:
# данные из колонок, названия которых заканчиваются на _str,
# считаются категориальными и подвергаются one hot encoderу;
# данные из колонки с датами (_date) разбиваются на праздники и выходные дни,
# дополнительно добавляется колонка с годом.
def preprocess_data(data):
    # Если нет target
    if 'Y' not in list(data):
        data['Y'] = [random.randint(0, 1) for _ in range(len(data))]
    # основные праздники календаря
    rus_holidays = {(datetime.date(1999, 1, 1), datetime.date(1999, 1, 8)): 'New_Year',
                    (datetime.date(1999, 2, 22), datetime.date(1999, 2, 23)): '23rd_of_February',
                    (datetime.date(1999, 3, 7), datetime.date(1999, 3, 8)): '8th_of_March',
                    (datetime.date(1999, 5, 1), datetime.date(1999, 5, 2)): 'Labor_Day',
                    (datetime.date(1999, 5, 8), datetime.date(1999, 5, 9)): 'Victory_Day',
                    (datetime.date(1999, 6, 11), datetime.date(1999, 6, 12)): 'Day_of_Russia',
                    (datetime.date(1999, 11, 3), datetime.date(1999, 11, 4)): 'National_Unity_Day'}

    for column in list(data):
        if column[-4:] == '_str':
            new_cols = pd.get_dummies(data[column])
            data = data.drop(column, axis=1)
            data = data.join(new_cols)
        elif column[-5:] == '_date':
            years = []
            meaning_of_day = []
            for row in data[column]:

                year, month, day = row.split('-')
                years.append(int(year))  # запомним год
                current_date = datetime.date(int(year), int(month), int(day)) # преобразуем строку с датой в datetime
                label = None

                # проверим на вхождение текущей даты в праздничные дни
                for date, name in rus_holidays.items():
                    date_begin = date[0]
                    date_end = date[1]
                    if date_begin.month <= current_date.month <= date_end.month and date_begin.day <= current_date.day <= date_end.day:
                        label = name

                 # если это никакой не праздник, то проверим, не выходной ли
                if not label:
                    if current_date.weekday() > 4:
                        label = 'Day_off'

                # запишем лейбл (либо название праздника, либо Day_off(выходной))
                meaning_of_day.append(label)

            data = data.drop([column], axis=1)

            data['Year'] = years
            # запишем колонку с лейблами и преобразуем в OneHot
            data['New'] = meaning_of_day
            new_cols = pd.get_dummies(data['New'])
            data = data.drop(['New'], axis=1)
            data = data.join(new_cols)

    return data


# Методы построения модели предсказаний
CLASSIFICATION = 'C'
REGRESSION = 'R'
YEAR_TASK = 'Y'
PATHS_TO_LISTS = 'P'


class BadPredictionModelMode(Exception):
    """
        Ошибка отражающая не верный параметр классификации
    """
    def __init__(self, message: str = ''):
        super().__init__()
        self.message = message


def make_model(model_specs):
    data = preprocess_data(model_specs)

    if model_specs['mode'] is CLASSIFICATION:
        model = RandomForestClassifier()
    elif model_specs['mode'] is REGRESSION:
        model = RandomForestRegressor()
    else:
        model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=500)
    # сохраняю модель тут
    X = data.drop(model_specs['target'], axis=1)
    y = data[model_specs['target']]
    model.fit(X, y)
    # вернуть id моделей, какая задача решалась ('mode'), id файла с данными
    return model, list(X)


def make_predict_model(file_id: str, classification_mode: str, task_mode: str, path_to_data):
    data = preprocess_data(pd.read_csv(path_to_data))  # предобратомаем данные
    # data = preprocess_data(pd.read_csv('purchase_test.csv')) -- для теста
    if task_mode is YEAR_TASK:  # для задачи по годам мы строим несколько моделей
        # мы требуем от пользователя, чтобы дата выглядела следующим образом 2016-12-23
        # тем самым обеспечиваем себе наличие хотя бы одного года в списке
        # может стоит сделать ограничение на выбор данной функции? чтоб проверялось наличие столбца с датой
        uniq_years = data['Year'].unique()
        result = []
        for year in np.sort(uniq_years):
            df = data[data['Year'] == year]
            model_ = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
            X = df.drop('Y', axis=1)
            y = df['Y']
            model_.fit(X, y)
            name = f"{file_id}_{classification_mode}_{task_mode}_{year}"
            result.append((model_, name))
        return result, list(X)

    if classification_mode is CLASSIFICATION:  # для всего остального нужна одна модель
        if task_mode is PATHS_TO_LISTS:
            model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
        else:
            model = RandomForestClassifier()
    elif classification_mode is REGRESSION:
        model = RandomForestRegressor()
    else:
        raise BadPredictionModelMode(
            'Не верный классификатор. Выберите один из {}'.format(', '.join([CLASSIFICATION, REGRESSION]))
        )

    X = data.drop(['Y'], axis=1)
    y = data['Y']
    model.fit(X, y)  # построим модель
    name = f'{file_id}_{classification_mode}_{task_mode}'
    return [(model, name)], list(X)


def rank_features(model, ft_names):
    ft_importances = model.feature_importances_
    indices = np.argsort(ft_importances)[::-1]
    ft_ranking = [(ft_names[i], ft_importances[i]) for i in indices]
    return ft_ranking


def tweak(path_to_data, model, groups):
    X = preprocess_data(pd.read_csv(path_to_data))  # предобратомаем данные
    y = X['Y'].mean()
    X.drop('Y', axis=1, inplace=True)

    predictions = []

    for group_cmb in chain.from_iterable(combinations(groups.keys(), k + 1) for k in range(len(groups))):
        X_copy = X.copy()
        pred = [y] + [0] * 10
        for i in range(10):
            for gr in group_cmb:
                for feat, step in groups[gr].items():
                    X_copy[feat] += step
            pred[i + 1] = model.predict(X_copy).mean()
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.plot(pred, marker='o')
        plt.title(' + '.join(group_cmb))
        plt.xticks(range(11))
        plt.xlabel('steps')
        plt.ylabel('target')
        plt.grid()
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.clf()
        predictions.append({
            'groups': list(group_cmb),  # list[{feature, step}] группы
            'pred': pred,  # list[int] список предсказанных 10 чисел
            'plot': str(base64.b64encode(img.getvalue()).decode())  # график b'str'
        })
    return predictions


def parsing_model_tree(model):
    children_left = model.tree_.children_left  # номера левых детей
    children_right = model.tree_.children_right  # номера правых детей
    feature = model.tree_.feature  # индексы фич (X_? <= 3.45)
    threshold = model.tree_.threshold  # пороги (X_1 <= ?)
    value = model.tree_.value  # список распределений классов для каждой вершины
    parent_id = [-1 for _ in range(len(value))]  # массив индексов родителей для каждой вершины - изначально -1

    # Найдем для каждой вершины родителя parent_id,
    # а так же запомним все листы, для которых будем запоминать
    # максимальное количество примеров в одном классе и сам класс
    stack = [(0, -1)]
    # далее для отрисовки понадобиться знать, в какую сторону идем в пути. Создадим этот массив на данном этапе:
    # если выполняется неравенство, то идем налево, и side для текущей вершины = True (пришли по True ветке)
    # иначе side = False
    sides = [False for _ in range(len(value))]
    max_leaf = Counter()
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            parent_id[children_left[node_id]] = node_id
            sides[children_left[node_id]] = True

            stack.append((children_right[node_id], parent_depth + 1))
            parent_id[children_right[node_id]] = node_id
            sides[children_right[node_id]] = False
        else:
            max_leaf[node_id] = {'value': np.max(value[node_id][0]) / np.sum(value[node_id][0]),
                                 'class': np.argmax(value[node_id][0])}

    # отсортируем полученные листы по не возрастанию
    important_nodes = [key for key in sorted(max_leaf, key=lambda x: max_leaf[x]['value'], reverse=True)]

    return important_nodes, parent_id, value, sides, feature, threshold


# функция, которая записывает в словарь информацию о левой и правой вершинах,
# а так же о ребре между ними и предыдущей
def add_node_vert(graph, prev, left, right, side, name, threshold, class_win, other_class):

    if side:
        if class_win:
            graph['nodes'].append({'ind': left, 'class_win': class_win, 'other_class': other_class})
        else:
            graph['nodes'].append({'ind': left, 'name': name, 'threshold': threshold, 'other_class': other_class})
        graph['nodes'].append({'ind': right})

        graph['verges'].append({'from': prev, 'to': right, 'side': True, 'color': 'coral1'})
        graph['verges'].append({'from': prev, 'to': left, 'side': True, 'color': 'palegreen3'})

    elif not side:
        graph['nodes'].append({'ind': left})
        if class_win:
            graph['nodes'].append({'ind': right, 'class_win': class_win, 'other_class': other_class})
        else:
            graph['nodes'].append({'ind': right, 'name': name, 'threshold': threshold, 'other_class': other_class})

        graph['verges'].append({'from': prev, 'to': left, 'side': False, 'color': 'coral1'})
        graph['verges'].append({'from': prev, 'to': right, 'side': False, 'color': 'palegreen3'})

    return graph


def visualize_results_after_prediction(model, feature_names):
    images = []
    names = []
    texts = []
    # У нас имеется построенное решающее дерево.
    # Мы хотим показать пользователю пути до важных вершин.
    # Для начала разберем структуру, которую получили после обучения от sklearn
    # за это отвечает ф-я parsing_model_tree
    important_nodes, parent_id, value, sides, feature, threshold = parsing_model_tree(model)
    labels_for_class = dict.fromkeys(range(len(value[0][0])), ord('a'))  # буквы в названиях картинок Class_1_?.png

    # построим граф для каждого важного листа
    for node in important_nodes:

        # найдем путь до этого листа, зная родителей вершин
        path = [node]
        while parent_id[node] != -1:
            path.append(parent_id[node])
            node = parent_id[node]

        nodes = path[::-1]
        prev = None
        prev_node_id = 0
        node_num = 1

        # нарисуем граф, сгенерировав dot файл
        template = Template('''
            digraph G {
            {
            node [style="rounded, filled" fontcolor=black fontsize=10 shape=ellipse]; {% for node in nodes %} {% if node.name %} 
            {{ node.ind }} [color=palegreen3, label=<<FONT POINT-SIZE="14" ><b>{{ node.name }} &lt;= {{ node.threshold }}<br align="left"/></b></FONT><FONT POINT-SIZE="12">{% for class in node.other_class %}Class {{ class.num }}: {{ class.sample }} samples ({{ class.percent }}%)<br align="left" />{% endfor %}</FONT>>]; {% elif node.class_win %} 
            {{ node.ind }} [color=palegreen3, label=<<FONT POINT-SIZE="14" ><b>Class {{ node.class_win }}<br align="left"/></b></FONT><FONT POINT-SIZE="12">{% for class in node.other_class %}Class {{ class.num }}: {{ class.sample }} samples ({{ class.percent }}%)<br align="left" />{% endfor %}</FONT>>]; {% else %} 
            {{ node.ind }} [color=coral1, label=""]; {% endif %} {% endfor %}
            } {% for verge in verges %} {% if verge.side %}
            {{ verge.from }} -> {{ verge.to }} [arrowhead=onormal, color={{ verge.color }}, headlabel="", labeldistance=2.5 labelangle=30]; {% else %}
            {{ verge.from }} -> {{ verge.to }} [arrowhead=onormal, color={{ verge.color }}, label=""]; {% endif %} {% endfor %}
            }
            ''')

        graph = {'nodes': [], 'verges': []}
        print_string = ['If']
        for node_id in nodes:
            side = sides[node_id]
            # подсчитаем распределение классов в данной вершине
            v = value[node_id]
            class_dict = []
            for i in range(len(value[node_id][0])):
                class_dict.append({'num': i + 1,
                                   'sample': int(v[0][i]),
                                   'percent': round(v[0][i] * 100 / np.sum(v), 1)})

            if not prev:  # если вершина - это корень
                # записываем в граф
                graph['nodes'].append({'ind': node_num, 'name': feature_names[feature[node_id]], 'threshold': round(threshold[node_id], 2), 'other_class': class_dict})
                # обновляем информацию о предыдущей вершины
                prev = node_num
                prev_node_id = node_id
                node_num += 1

            else:  # все оставшиеся вершины
                # обновляем номера вершин левой и правой сторон
                left = node_num
                right = node_num + 1
                node_num += 2

                if nodes[-1] != node_id:  # если это не лист
                    # записываем в граф
                    name = feature_names[feature[node_id]]
                    threshold_ = round(threshold[node_id], 2)
                    graph = add_node_vert(graph, prev, left, right, side, name, threshold_, None, class_dict)
                    # генерируем текстовый путь до листа
                    sep = '<=' if side else '>'
                    print_string.append(f"{feature_names[feature[prev_node_id]]} {sep} {round(threshold[prev_node_id], 2)} and")
                    # обновляем информацию о предыдущей вершины
                    if side:
                        prev = left
                    else:
                        prev = right
                    prev_node_id = node_id
                else:  # все же лист
                    winner = int(np.argmax(value[node_id][0])) + 1
                    # записываем в граф
                    graph = add_node_vert(graph, prev, left, right, side, None, None, winner, class_dict)
                    num_of_class = int(np.argmax(value[node_id][0]))  # для названия вкладки сохраняем класс
                    # генерируем текстовый путь до листа
                    sep = '<=' if side else '>'
                    print_string.append(
                        f"{feature_names[feature[prev_node_id]]} {sep} {round(threshold[prev_node_id], 2)}")
                    print_string.append(f'then Class {int(np.argmax(value[node_id][0])) + 1}')

        # записываем всю ин-цию о пути в шаблон
        s = template.render(graph)
        with open('images/graph.dot', 'w') as f:
            f.write(s)
        # переводим в формат png
        # num_of_class объявление переменной будет в любом случае
        # nodes[-1] != node_id для каждого пути будет один раз в конце False, когда мы дойдем до листа,
        # и именно оттуда вытащим, какой это класс
        file = f'Class_{num_of_class + 1}_{chr(labels_for_class[num_of_class])}.png'
        os.system(f'dot -Tpng images/graph.dot -o images/{file}')
        # записываем картинку, текст к ней и заголовок для вкладки
        labels_for_class[num_of_class] += 1

        # сконвертируем картинки в байты и запишем на выход
        # для обратного преображения: base64.b64decode(image)
        with open(f"images/{file}", "rb") as imageFile:
            img_to_str = base64.b64encode(imageFile.read())
        images.append(str(img_to_str.decode()))

        # path = os.path.abspath(os.path.join(os.path.dirname(__file__)) + f'/images/{file}')
        # pics.append(path)  # сами картинки
        names.append(file.split('.')[0])  # имена вкладок
        texts.append(' '.join(print_string))  # текст к картинкам

    return images, names, texts


def visual_tree(dtree, name, ft_names):
    dot_data = os.path.join(f"trees/{name}.dot")
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=ft_names)
    os.system(f"dot -Tpng trees/{name}.dot -o trees/{name}.png")
    # return os.path.abspath(os.path.join(os.path.dirname(__file__)) + f'/trees/{name}.png')
    with open(f"trees/{name}.png", "rb") as imageFile:
        img_to_str = base64.b64encode(imageFile.read())
    return str(img_to_str.decode()), name.split('_')[3].split('.')[0]


def trees_for_years(model_specs):
    data = preprocess_data(model_specs)
    # разобьем данные по годам
    # и отрисуем деревья для каждого года с помощью export_graphviz
    uniq_years = data['Year'].unique()
    trees = []
    for year in np.sort(uniq_years):
        df = data[data['Year'] == year]
        model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
        X = df.drop('Y', axis=1)
        y = df['Y']
        model.fit(X, y)
        path = visual_tree(model, f'{year}_year', list(X))
        trees.append(path)
    return trees
