import catboost as cb
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from sklearn.metrics import mean_squared_error as mse

from option_pricing import BlackScholes as bs
from option_pricing import differentialML as dfml
from option_pricing import riskFuel as rf

import pickle
import os

app = Flask(__name__)
api = Api(app)

model_path = 'saved_models/'

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
    1 : f"{model_path}model1.pkl",
    2: f"{model_path}model2.cbm"
}


@app.route('/', methods=['GET', 'POST'])
def list_of_models():
    """Returns dict {model_id: model_name} with models, available for training and testing"""
    # response = {'Availiable models': {1: 'RiskFuel', 2: 'CatBoost', 3: 'DiffML'}}
    response = {'Availiable models': {1: 'RiskFuel', 2: 'CatBoost'}}
    return jsonify(response)


@app.route('/saved_models', methods=['GET'])
def list_of_saved_models():
    """Returns a list of filenames for trained and saved models.
    The num in filename corresponds to the model id."""
    response = {'Saved models': os.listdir(model_path)}
    return jsonify(response)


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
        response = {'model name': 'RiskFuel',
                    'params description': {'n_hidden': 'non-negative integer, neurons per each linear layer',
                                           'n_layers': 'non-negative integer, num of linear layers',
                                           'n_epochs': 'non-negative integer, num of training epochs'},
                    'default params': model_params['RiskFuel'],
                    }
        print(jsonify(response))
        return jsonify(response)

    def post(self):
        params = request.json
        if 'n_hidden' not in params:
            params.update({'n_hidden': 100})
        if 'n_layers' not in params:
            params.update({'n_layers': 3})
        return params


api.add_resource(RiskFuel, '/1')


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
        response = {'model name': 'CatBoost',
                    'params description': {'lr': 'positive real, learning rate',
                                           'depth': 'positive integer, max depth for the tree',
                                           'iterations': 'positive integer, num of iterations'},
                    'default params': model_params['CatBoost'],
                    }
        print(jsonify(response))
        return jsonify(response)

    def post(self):
        params = request.json
        model_params['CatBoost'].update(dict(params))
        return jsonify(model_params['CatBoost'])


api.add_resource(CatBoost, '/2')


class DiffML(Resource):
    """
        A class to represent DiffML model.

        Methods
        -------
        get:
            To see the description of the model, parameters description and default parameters
        post:
            Update model parameters according to a given input
    """
    def get(self):
        response = {'model name': 'DiffML',
                    'params description': {'differential': 'boolean, depending on what model you would like to use',
                                           'lambda': 'non-negative real number, parameter used in loss function'
                                                     'MSE_val + lambda * MSE_deriv'},
                    'default params': model_params['DiffML'],
                    }
        return jsonify(response)

    def post(self):
        params = request.json
        if 'differential' not in params:
            params.update({'differential': True})
        if 'lambda' not in params:
            params.update({'lambda': 1})
        return params


api.add_resource(DiffML, '/3')


@app.route('/train_set_params', methods=['GET', 'POST'])
def set_train_generator_params():
    """Set parameters for generator to further create training and testing sets"""
    global gen_params, generator

    if request.method == 'GET':
        response = {'default params for training set generator': gen_params}
    if request.method == 'POST':
        response = gen_params
        params = dict(request.json)
        for p in params:
            if p not in response:
                raise Exception(f"You can't specify parameter {p}. "
                                f"You can only specify parameters {set(gen_params.keys())}")
            if p in {"spot", "time", "sigma", "rate"}:
                try:
                    a, b = params[p]
                except:
                    Exception(f"Parameter {p} must be a pair (begin, end), that specifies the range to sample from")
                if a >= b:
                    Exception(f"Parameter {p} must be a pair (begin, end), where begin < end")
        response.update(params)

    gen_params = response
    generator = bs.DataGen(gen_params['spot'], gen_params['time'],
                           gen_params['sigma'], gen_params['rate'])
    return jsonify(response)


@app.route('/train_model/<int:model_id>', methods=['GET', 'POST'])
def train_model(model_id):
    """
    Train and save model for given model id. If the model has been already trained, it will be overwritten
    :param model_id: 1 for RiskFuel, 2 for CatBoost
    :return: dict with report. dict keys: ["status", "model score", "model params", "model path"].
    "status" shows if the model was successfully triained
    "model score" shows MSE on test set
    "model params" shows params the model was trained with (default or specified using POST on http://127.0.0.1:5001/model_id)
    "model path" shows path to the trained and saved model
    """
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

    if model_id == 3:
        prms = model_params['DiffML']
        differential, lam = prms['differential'], prms['lambda']
        regressor = dfml.Neural_Approximator(xTrain, yTrain, dydxTrain)
        print("done")

        regressor.prepare(gen_params['train_size'], differential, weight_seed=42, lam=lam)
        regressor.train(xTrain, yTrain, dydxTrain, "differential training", lam=lam)
        predvalues, preddiff = regressor.predict_values_and_derivs(xTest)
        mse_on_test = mse(yTest, predvalues)
        # todo: надо научиться сохранять веса модели в pickle
        return "Saving model weights for this option is not implemented yet. Please, try another model"

    raise Exception("You need to specify correct model id. "
                    "Check http://127.0.0.1:5001/ to see the ids for available models")


def del_file(path):
    try:
        os.remove(path)
    except IOError:
        Exception("No such file. Check http://127.0.0.1:5001/saved_models to see the list of saved models")


@app.route('/delete/<int:model_id>', methods=['DELETE'])
def delete_saved_model(model_id):
    """
    Delete saved model for given model id
    :param model_id: 1 for RiskFuel, 2 for CatBoost
    :return deletion status
    """
    if model_id == 1:
        del_file(path_to['RiskFuel'])
        return jsonify(f"Model {model_id} on path {path_to['RiskFuel']} "
                       f"has been successfully deleted!")
    elif model_id == 2:
        del_file(path_to['CatBoost'])
        return jsonify(f"Model {model_id} on path {path_to['CatBoost']} "
                       f"has been successfully deleted!")
    raise Exception("You need to specify correct model id."
                    " Check http://127.0.0.1:5001/ to see the ids for available models"
                    " and http://127.0.0.1:5001/saved_models to see the list of saved models")

@app.route('/predict/<int:model_id>', methods=['POST'])
def predict(model_id):
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
                       f" Use http://127.0.0.1:5001/train_model/{model_id} "
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
        return jsonify({"model_id": 1,
                        "model_name": 'RiskFuel',
                        "model_params": model_params['RiskFuel'],
                        "prediction": list(prediction)})
    if model_id == 2:
        model = cb.CatBoostRegressor()
        model.load_model(path)
        prediction = model.predict(data).astype(float)
        return jsonify({"model_id": 2,
                        "model_name": 'CatBoost',
                        "model_params": model_params['CatBoost'],
                        "prediction": list(prediction)})
    raise Exception("You need to specify correct model id."
                    " Check http://127.0.0.1:5001/ to see the ids for available models"
                    " and http://127.0.0.1:5001/saved_models to see the list of saved models")
if __name__ == '__main__':
    app.run(debug=True, port=5001)
