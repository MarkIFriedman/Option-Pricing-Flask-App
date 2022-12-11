import catboost as cb
import numpy as np
from flask import Flask, jsonify, request
from flask_restx import Api, Resource
from werkzeug.exceptions import BadRequest
from sklearn.metrics import mean_squared_error as mse

from option_pricing import BlackScholes as bs
from option_pricing import riskFuel as rf

from flask_sqlalchemy import SQLAlchemy

import pickle
import os

app = Flask(__name__)
# app.config.from_object("project.config.Config")
api = Api(app)
db = SQLAlchemy(app)

model_path = 'saved_models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

gen_params = {
    'train_size': 1000,
    'test_size': 300,
    "spot": (0.5, 2),
    "time": (0, 3.0),
    "sigma": (0.1, 0.5),
    "rate": (-0.01, 0.03)
}
generator = bs.DataGen(gen_params['spot'], gen_params['time'],
                       gen_params['sigma'], gen_params['rate'])

model_params = {
    'CatBoost': {'depth': 3, 'lr': 0.1, 'iterations': 200},
    'DiffML': {'differential': True, 'lambda': 1},
    'RiskFuel': {'n_hidden': 100, 'n_layers': 3, 'n_epochs': 200}
}

path_to = {
    'RiskFuel': f"{model_path}model1.pkl",
    'CatBoost': f"{model_path}model2.cbm",
    1: f"{model_path}model1.pkl",
    2: f"{model_path}model2.cbm"
}


# @app.route('/list_of_models', methods=['GET', 'POST'])
@api.route('/list_of_models', endpoint='models', methods=['POST'])
class ListOfModels(Resource):
    def get(self):
        """Returns dict {model_id: model_name} with models, available for training and testing"""
        # response = {'Availiable models': {1: 'RiskFuel', 2: 'CatBoost', 3: 'DiffML'}}
        response = {'Availiable models': {1: 'RiskFuel', 2: 'CatBoost'}}
        return response


api.add_resource(ListOfModels, '/list_of_models')


# @app.route('/saved_models', methods=['GET'])
@api.route('/saved_models', endpoint='saved_models', methods=['GET'])
class ListOfSavedModels(Resource):
    def get(self):
        """Returns a list of filenames for trained and saved models.
        The num in filename corresponds to the model id."""
        response = {'Saved models': os.listdir(model_path)}
        return jsonify(response)


api.add_resource(ListOfSavedModels, '/saved_models')

@api.route('/RiskFuel',  endpoint='1', methods=['GET', 'POST'])
class RiskFuel(Resource):
    """
    A class to represent RiskFuel model.

    Methods
    -------
    get:
        To see the description of the model, parameters description and default parameters
    post:
        Update model parameters according to a given input
    """

    def get(self):
        """To see the description of the model, parameters description and default parameters"""
        response = {'model name': 'RiskFuel',
                    'model_id': 1,
                    'params description': {'n_hidden': 'non-negative integer, neurons per each linear layer',
                                           'n_layers': 'non-negative integer, num of linear layers',
                                           'n_epochs': 'non-negative integer, num of training epochs'},
                    'default params': model_params['RiskFuel'],
                    }
        print(jsonify(response))
        return jsonify(response)

    @api.doc(params={'n_hidden': {'description': 'num of neurons per each linear layer',
                                  'type': int, 'default': model_params['RiskFuel']['n_hidden']},
                     'n_layers': {'description': 'num of linear layers',
                                  'type': int, 'default': model_params['RiskFuel']['n_layers']},
                     'n_epochs': {'description': 'num of training epochs',
                                  'type': int, 'default': model_params['RiskFuel']['n_epochs']},
                     },
             responses={200: 'Model params has been successfully set',
                        400: 'Bad request'}
             )
    def post(self):
        """Update model parameters according to a given input"""
        params = dict(request.args)
        params = {k: int(v) for k, v in params.items()}
        model_params['RiskFuel'].update(params)
        return jsonify(model_params['RiskFuel'])


# api.add_resource(RiskFuel, '/RiskFuel')

@api.route('/CatBoost',  endpoint='2', methods=['GET', 'POST'])
class CatBoost(Resource):
    """
        A class to represent CatBoost model.

        Methods
        -------
        get:
            To see the description of the model, parameters description and default parameters
        post:
            Update model parameters according to a given input
    """

    def get(self):
        """To see the description of the model, parameters description and default parameters"""
        response = {'model name': 'CatBoost',
                    'model id': 2,
                    'params description': {'lr': 'positive real, learning rate',
                                           'depth': 'positive integer, max depth for the tree',
                                           'iterations': 'positive integer, num of iterations'},
                    'default params': model_params['CatBoost'],
                    }
        print(jsonify(response))
        return jsonify(response)

    @api.doc(params={'lr': {'description': 'learning rate for gradient descent',
                                  'type': float, 'default': model_params['CatBoost']['lr']},
                     'depth': {'description': 'max depth for the tree',
                                  'type': int, 'default': model_params['CatBoost']['depth']},
                     'iterations': {'description': 'num of iterations for training',
                                  'type': int, 'default': model_params['CatBoost']['iterations']},
                     },
             responses={200: 'Model params has been successfully set',
                        400: 'Bad request'}
             )
    def post(self):
        """Update model parameters according to a given input"""
        params = dict(request.args)
        params1 = {k: int(v) for k, v in params.items() if k != "lr"}
        if "lr" in params.keys():
            params1["lr"] = float(params["lr"])
        model_params['CatBoost'].update(params1)
        return jsonify(model_params['CatBoost'])


# api.add_resource(CatBoost, '/CatBoost')


# class DiffML(Resource):
#     """
#         A class to represent DiffML model.
#
#         Methods
#         -------
#         get:
#             To see the description of the model, parameters description and default parameters
#         post:
#             Update model parameters according to a given input
#     """
#
#     def get(self):
#         """See the description of the model, parameters description and default parameters"""
#         response = {'model name': 'DiffML',
#                     'model id': 3,
#                     'params description': {'differential': 'boolean, depending on what model you would like to use',
#                                            'lambda': 'non-negative real number, parameter used in loss function'
#                                                      'MSE_val + lambda * MSE_deriv'},
#                     'default params': model_params['DiffML'],
#                     }
#         return jsonify(response)
#
#     def post(self):
#         """Update model parameters according to a given input"""
#         params = request.json
#         if 'differential' not in params:
#             params.update({'differential': True})
#         if 'lambda' not in params:
#             params.update({'lambda': 1})
#         return params
#
#
# api.add_resource(DiffML, '/DiffML')

def str_to_list(s):
    return list(map(float, s.split(', ')))


@api.route('/train_set_params', methods=['GET', 'POST'])
class SetTrainGeneratorParams(Resource):
    def get(self):
        """Show current parameters for training set generator"""
        response = {'current generator params': gen_params}
        return jsonify(response)

    @api.doc(params={'train_size': {'description': f'num objects for training dataset',
                                    'type': int, 'default': gen_params['train_size']},
                     'test_size': {'description': f'num objects for testing dataset. ',
                                   'type': int, 'default': gen_params['test_size']},
                     'spot': {'description': f'range for spot parametr for generator . '
                                             f'Two float values comma separated',
                              'type': str, 'default': "0.5, 2."},
                     'time': {'description': f'range for time parametr for generator. '
                                             f'Two float values comma separated',
                              'type': str, 'default': "0, 3"},
                     'sigma': {'description': f'range for sigma parametr for generator. '
                                              f'Two float values comma separated',
                               'type': str, 'default': "0.1, 0.5"},
                     'rate': {'description': f'range for rate parametr for generator. '
                                             f'Two float values comma separated',
                              'type': str, 'default': "-0.01, 0.03"}})
    @api.doc(responses={200: 'Training set params has been successfully set',
                        400: 'Bad request'})
    def post(self):
        """Set parameters for generator to further create training and testing sets"""
        global gen_params, generator
        response = gen_params
        params = dict(request.args)
        for p in params:
            if p not in response:
                raise BadRequest(f"You can't specify parameter {p}. "
                                 f"You can only specify parameters {set(gen_params.keys())}")
            if p in {"spot", "time", "sigma", "rate"}:
                try:
                    params[p] = str_to_list(params[p])
                except:
                    raise BadRequest(f"Wrong input format for parameter {p}. Must two float values comma separated")
                try:
                    a, b = params[p]
                except:
                    raise BadRequest(
                        f"Parameter {p} must be a pair (begin, end), that specifies the range to sample from")
                if a >= b:
                    raise BadRequest(f"Parameter {p} must be a pair (begin, end), where begin < end")
            else:
                params[p] = int(params[p])
        response.update(params)

        gen_params = response
        generator = bs.DataGen(gen_params['spot'], gen_params['time'],
                               gen_params['sigma'], gen_params['rate'])
        return jsonify(response)


# api.add_resource(SetTrainGeneratorParams, '/train_set_params')


@api.route('/train_model', endpoint='train', methods=['POST'])
class TrainModel(Resource):
    @api.doc(params={'model_id': {'description': 'model id to be trained: 1 for RiskFuel, 2 for CatBoost',
                                  'type': int, 'default': 1}},
             responses={200: 'Model has been successfully trained and saved',
                        400: 'Bad request'}
             )
    def post(self):
        """
        Train and save model for given model id. If the model has been already trained, it will be overwritten
        :param model_id: 1 for RiskFuel, 2 for CatBoost
        :return: dict with report. dict keys: ["status", "model score", "model params", "model path"].
        "status" shows if the model was successfully triained
        "model score" shows MSE on test set
        "model params" shows params the model was trained with (default or specified using POST method for a specific model)
        "model path" shows path to the trained and saved model
        """
        try:
            model_id = int(request.args.get('model_id'))
        except BadRequest:
            return 'model_id must be integer'
        xTrain, yTrain, dydxTrain = generator.dataset(gen_params['train_size'], seed=42)
        xTest, yTest, dydxTest = generator.dataset(gen_params['test_size'], seed=43)
        if model_id == 1:
            params = model_params['RiskFuel']
            net = rf.RiskFuelNet(n_feature=4, n_hidden=params['n_hidden'],
                                 n_layers=params['n_layers'], n_output=1)
            n_epochs = params['n_epochs']
            ls, checkpoint, l_train, l_test = rf.fit_net(net, n_epochs, xTrain, yTrain,
                                                         xTest, yTest)

            with open(path_to['RiskFuel'], 'wb') as handle:
                pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

            score = mse(rf.predict(net, xTest), yTest)
            response = {"status": "RiskFuel model has been successfully trained!",
                        "model score": score,
                        "model params": model_params['RiskFuel'],
                        "model path": path_to['RiskFuel']}

            return jsonify(response)
        if model_id == 2:
            test_dataset = cb.Pool(xTest, yTest)
            params = model_params['CatBoost']
            model = cb.CatBoostRegressor(iterations=params['iterations'],
                                         max_depth=params['depth'],
                                         learning_rate=params['lr'],
                                         random_seed=42,
                                         logging_level='Silent')
            model.fit(xTrain, yTrain, eval_set=test_dataset, use_best_model=True, early_stopping_rounds=10)

            model.save_model(path_to['CatBoost'])
            score = mse(model.predict(xTest), yTest)
            response = {"status": "CatBoost model has been successfully trained!",

                        "model score": score,
                        "model params": model_params['CatBoost'],
                        "model path": path_to['CatBoost']}
            return jsonify(response)

        # if model_id == 3:
        #     prms = model_params['DiffML']
        #     differential, lam = prms['differential'], prms['lambda']
        #     regressor = dfml.Neural_Approximator(xTrain, yTrain, dydxTrain)
        #     print("done")
        #
        #     regressor.prepare(gen_params['train_size'], differential, weight_seed=42, lam=lam)
        #     regressor.train(xTrain, yTrain, dydxTrain, "differential training", lam=lam)
        #     predvalues, preddiff = regressor.predict_values_and_derivs(xTest)
        #     score = mse(yTest, predvalues)
        #     # todo: надо научиться сохранять веса модели в pickle
        #     response = {"status": "DiffML model has been trained but wasn't saved."
        #                           " You can't use this model for further prediction",
        #
        #                 "model score": score,
        #                 "model params": model_params['DiffML'],
        #                 "model path": 'Saving model weights for this option is not implemented yet.'}
        #     return response

        raise BadRequest("You need to specify correct model id. "
                         "Check http://127.0.0.1:5001/list_of_models to see the ids for available models")


# api.add_resource(TrainModel, '/train_model')


def del_file(path):
    try:
        os.remove(path)
    except IOError:
        Exception("No such file. Check http://127.0.0.1:5001/saved_models to see the list of saved models")


@api.route('/delete', methods=['DELETE'])
class DeleteSavedModel(Resource):
    """
    Delete saved model for given model id
    :param model_id: 1 for RiskFuel, 2 for CatBoost
    :return deletion status
    """

    @api.doc(params={'model_id': {'description': 'model id to be trained: 1 for RiskFuel, 2 for CatBoost',
                                  'type': int, 'default': 1}},
             responses={200: 'Model has been successfully deleted',
                        400: 'Bad request'}
             )
    def delete(self):
        print(request.args)
        try:
            model_id = int(request.args.get('model_id'))
        except BadRequest:
            return 'model_id must be integer'
        if model_id == 1:
            del_file(path_to['RiskFuel'])
            return jsonify(f"Model {model_id} on path {path_to['RiskFuel']} "
                           f"has been successfully deleted!")
        elif model_id == 2:
            del_file(path_to['CatBoost'])
            return jsonify(f"Model {model_id} on path {path_to['CatBoost']} "
                           f"has been successfully deleted!")
        raise BadRequest("You need to specify correct model id."
                        " Check http://127.0.0.1:5001/list_of_models to see the ids for available models"
                        " and http://127.0.0.1:5001/saved_models to see the list of saved models")


data_schema = {
    'type': 'object',
    'properties': {
        'rate': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'sigma': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'spot': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'time': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },

    },
    'additionalProperties': False,
    'required': ['rate', 'sigma', 'spot', 'time'],
}
request_data = api.schema_model('data', data_schema)

@api.route('/predict/<int:model_id>', methods=['POST'])
class Predict(Resource):
    @api.doc(params={'model_id': {'description': 'model id to be trained: 1 for RiskFuel, 2 for CatBoost',
                                  'type': int, 'default': 1}})
    @api.expect(request_data, validate=True)
    def post(self, model_id):
        """
        Predict option prices for given model id and input parameters
        :param model_id: 1 for RiskFuel, 2 for CatBoost
        input parameters format: dict with keys ["spot", "time", "sigma", "rate"] and list values
        input example:
            {
            "rate": [1, 5, 2, 1],
            "sigma": [2, 1, 3, 2],
            "spot": [1, 3 ,2, 1],
            "time": [8, 7, 6, 4]
            }
        :return dict with model description and prediction
        """
        data = request.json
        features = ["spot", "time", "sigma", "rate"]
        empty_inp = {f: [] for f in features}
        try:
            data = [np.array(data[f]) for f in features]
        except:
            return jsonify({f"Wrong input format. Input must be the following dictionary":
                                empty_inp})
        data = np.vstack(data).T
        path = path_to[model_id]
        if not os.path.isfile(path):
            return jsonify(f"File {path} does not exist."
                           f"You need to train this model first."
                           f" Use POST method on train_model with model_id={model_id} "
                           f"for training the model.")
        if model_id == 1:
            with open(path, 'rb') as f:
                params = pickle.load(f)
            model = rf.RiskFuelNet(n_feature=4,
                                   n_hidden=params["n_hidden"],
                                   n_layers=params["n_layers"],
                                   n_output=1)
            model.load_state_dict(params['model_state_dict'])
            model.eval()
            model.to(rf.device)
            prediction = rf.predict(model, data).astype(float)
            model_descr = {"id": 1,
                            "name": 'RiskFuel',
                            "params": model_params['RiskFuel']}
            return jsonify({"model" : model_descr,
                            "prediction": list(prediction)})
        if model_id == 2:
            model = cb.CatBoostRegressor()
            model.load_model(path)
            prediction = model.predict(data).astype(float)
            model_descr = {"id": 2,
                            "name": 'CatBoost',
                            "params": model_params['CatBoost']}
            return jsonify({"model" : model_descr,
                            "prediction": list(prediction)})
        raise Exception("You need to specify correct model id."
                        " Check http://127.0.0.1:5001/list_of_models to see the ids for available models"
                        " and http://127.0.0.1:5001/saved_models to see the list of saved models")


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
