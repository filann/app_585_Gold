import json
import os
import re
import time
import uuid
import pickle
import pandas as pd
from flask import (
    Flask,
    abort,
    request,
    make_response,
    jsonify,
    render_template,
    views,
    Response)
from werkzeug.utils import secure_filename
from tweakerAPI.coreML import (rank_features, make_predict_model, BadPredictionModelMode, tweak,
                               visual_tree, visualize_results_after_prediction)

UPLOAD_DIR = 'uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

VALID_FILENAME = re.compile('\S+\.json')

# Получение рабочей директории.
# Для корректной работы сервер нужно запускать из папки csc585
WORK_DIR = os.getcwd()
# Определение папки щаблонов
TEMPLATE_FOLDER = os.path.join(WORK_DIR, 'web-app', 'dist')
# Определение папки статики
STATIC_FOLDER = os.path.join(WORK_DIR, 'web-app', 'dist', 'static')
app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER,
)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)


@app.route('/tweaker/v2/data/upload', methods=['POST'])
def upload_user_data():
    """
        Загрузка файла с данными признаков для обучения модели.
        Сохраняется файл имя составляется из сгененрированного guid`a и оригиналбьного имени файла.
        Результатом успешного зароса является код 201 и идентификатор файла.
    """
    user_features_file = request.files.get('userFeaturesFile')
    if not user_features_file:
        abort(400)
    if not user_features_file.filename:
        abort(400)
    file_id = uuid.uuid4()
    filename = f'{file_id}_{secure_filename(user_features_file.filename)}'
    user_features_file.save(os.path.join(UPLOAD_DIR, filename))
    return jsonify(
        {
            'uuid': file_id,
        }
    ), 201


CLASSIFICATION = 'C'
REGRESSION = 'R'
YEAR_TASK = 'Y'
PATHS_TO_LISTS = 'P'
IMPORTANT_FEATURES = 'I'


@app.route('/tweaker/v2/data/processing', methods=['POST'])
def process_user_data():
    """
        Запуск процесса обработки данных
        Ожидается парметр выбора алгоритма обаботки
        С - алгорит классификации
        R - алгоритм регресии.

        Ожидается парметр выбора подзадачи
        O - отток клиентов
        I - важные фичи
        P - пути до листов
        Y - построение деревьев по годам
    """
    if not request.json:
        abort(400)

    request_json = json.loads(request.json)
    file_id = request_json.get('storedFileUid')
    classification_mode = request_json.get('classificationMode')
    task_mode = request_json.get('taskMode')

    # Пытаемся получить спиок из одного имени файла.
    one_target_file = [file_name
                       for file_name in os.listdir(UPLOAD_DIR)
                       if file_id == file_name.split('_')[0]]
    if not one_target_file:
        abort(Response('File not found'), status=404)
    one_target_file = one_target_file[0]
    # имя модели состоит из трех параметров:
    # {file_id}_{classification_mode}_{task_mode}_{дополнительный параметр(опционально)}.mdl
    # ищем нужную модель по параметрам задачи
    model_files = [file_name.split('.')[0]
                   for file_name in os.listdir(MODEL_DIR)
                   if (file_name.split('_')[0] == file_id and
                       file_name.split('_')[1] == classification_mode and
                       file_name.split('_')[2].split('.')[0] == task_mode)]
    # если модель не найдена, то строим и сохраняем
    if not model_files:
        try:
            path_to_data = os.path.join(UPLOAD_DIR, one_target_file)  # путь до данных
            models, feature_names = make_predict_model(file_id, classification_mode, task_mode, path_to_data)  # модель и наименования фич
            try:
                model_files = [name for model, name in models]  # список id моделей
                for model, name in models:  # сохраним модели
                    model_file_path = os.path.join(MODEL_DIR, name)
                    with open(model_file_path + '.mdl', 'wb') as file:
                        pickle.dump(model, file)

                feature_names_path = os.path.join(MODEL_DIR, str(file_id))  # сохраним названия фич
                with open(feature_names_path + '.fn', 'wb') as file:
                    pickle.dump(feature_names, file)

            except OSError:
                # TODO приделать логгер
                abort(Response('Prediction file not save'), status=500)

        except (FileNotFoundError, OSError):
            abort(Response('Find bad file'), status=424)
        except BadPredictionModelMode as err:
            abort(Response(err.message), status=400)

    return jsonify({'uuid': file_id,
                    'classificationMode': classification_mode,
                    'taskMode': task_mode,
                    'modelIds': model_files  # list(str)
                    }), 201


@app.route('/tweaker/v2/data/results', methods=['POST'])
def get_results():
    """
            Запуск процесса обработки данных
            Ожидается парметр выбора алгоритма обаботки
            С - алгорит классификации
            R - алгоритм регресии.

            Ожидается парметр выбора подзадачи
            O - отток клиентов
            ...
            P - пути до листов
            Y - построение деревьев по годам
    """
    if not request.json:
        abort(400)

    request_json = json.loads(request.json)
    file_id = request_json.get('storedFileUid')
    classification_mode = request_json.get('classificationMode')
    task_mode = request_json.get('taskMode')
    model_ids = request_json.get('modelIds')
    feature_name_id = file_id

    if not model_ids:
        abort(Response('Model file not found'), status=404)

    # названия фич одни для всех задач. загрузим их
    with open(os.path.join(MODEL_DIR, feature_name_id) + '.fn', 'rb') as file:
        feature_names = pickle.load(file)

    if task_mode is YEAR_TASK:
        images = []
        names = []
        for filename in model_ids:
            with open(os.path.join(MODEL_DIR, filename) + '.mdl', 'rb') as file:
                model = pickle.load(file)
            image, name = visual_tree(model, filename, feature_names)
            images.append(image)
            names.append(f'{name}_year')
        return jsonify({'uuid': file_id,
                        'classificationMode': classification_mode,
                        'taskMode': task_mode,
                        'treeImages': images,  # list(b'image') список картинок с деревьями (base64.b64decode(image))
                        'namesForTabs': names
                        }), 201
    else:
        try:

            # во всех остальных задачах модель одна, загрузим ее
            with open(os.path.join(MODEL_DIR, model_ids[0]) + '.mdl', 'rb') as file:
                model = pickle.load(file)

            if task_mode is PATHS_TO_LISTS:
                images, names_for_tabs, path_text = visualize_results_after_prediction(model, feature_names)
                return jsonify({'uuid': file_id,
                                'classificationMode': classification_mode,
                                'taskMode': task_mode,
                                'pathImages': images,  # list(b'image') список картинок с деревьями (base64.b64decode(image))
                                'namesForTabs': names_for_tabs,  # list(str) список названий для табов (Class_1_a, например)
                                'pathText': path_text  # list(str) список текстов, описывающих картинки
                                }), 201

            elif task_mode is IMPORTANT_FEATURES:
                feature_ranking = rank_features(model, feature_names)

                groups = request_json.get('groupNames')  # список словарей {фича, шаг}
                one_target_file = [file_name
                                   for file_name in os.listdir(UPLOAD_DIR)
                                   if file_id == file_name.split('_')[0]]
                if not one_target_file:
                    abort(Response('File not found'), status=404)
                one_target_file = one_target_file[0]
                try:
                    path_to_data = os.path.join(UPLOAD_DIR, one_target_file)
                    predictions = tweak(path_to_data, model, groups)

                    return jsonify({'uuid': file_id,
                                    'classificationMode': classification_mode,
                                    'taskMode': task_mode,
                                    'featureRanking': feature_ranking,  # [['x9', 0.1054252950573323], ['x1', 0.10136555217024072], ..]
                                    'predictions': predictions  # непростая вещь
                                    # это список словарей list(dicts), где каждый словарь состоит из:
                                    # 'groups': ['group1', 'group2'] список входящих групп
                                    # 'pred': list[float] список предсказанных 10 чисел
                                    # 'plot': график b'str'
                                    }), 201
                except (FileNotFoundError, OSError):
                    abort(Response('Find bad file'), status=424)
                except BadPredictionModelMode as err:
                    abort(Response(err.message), status=400)
        except OSError:
            # TODO приделать логгер
            abort(Response('Prediction file not save'), status=500)


class MainPageView(views.MethodView):
    """
        Предсталение рендеринга SPA
    """

    @staticmethod
    def get():
        """
            Получение веб приложения
        """
        return render_template('index.html')


if __name__ == '__main__':
    app.add_url_rule('/', view_func=MainPageView.as_view('web_face'))
    app.run(port=8000, debug=True)
