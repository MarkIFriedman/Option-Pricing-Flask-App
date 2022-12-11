from flask import Flask, jsonify, request, render_template
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/', methods=['GET', 'POST'])
def list_of_models():
    return render_template("list_of_models.html")


class DiffML(Resource):
    def get(self):
        return {'hello': 'world'}
api.add_resource(DiffML, '/diffml')


class RiskFuel(Resource):
    def get(self):
        return {'hello': 'world'}
api.add_resource(RiskFuel, '/riskfuel/')




@app.route('/trainig_set_params/', methods=['POST'])
def my_form_post():
    return render_template("set_params.html")
    # if request.form.get('RiskFuel') == 'RiskFuel':
    #     return render_template("set_params.html")
    # if request.form.get('DiffML') == 'Differential ML':
    #     return render_template("set_params.html")




if __name__ == '__main__':
    app.run(debug=True, port=5005)
