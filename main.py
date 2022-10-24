from flask import Flask, jsonify, request, render_template
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/', methods=['GET'])
def list_of_models():
    return render_template("list_of_models.html")


class DiffML(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(DiffML, '/')


class RiskFuel(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(RiskFuel, '/')


@app.route('/')
def input_form():
    return render_template("input_form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    if request.form.get('RiskFuel') == 'RiskFuel':
        return render_template("set_params.html")
    if request.form.get('DiffML') == 'Differential ML':
        return {'diff': 'ml'}



if __name__ == '__main__':
    app.run(debug=True, port=5005)
