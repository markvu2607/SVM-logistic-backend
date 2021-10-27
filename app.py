from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from svm import modelSVC
from logistic import modelLogistic
from sklearn.metrics import accuracy_score
import pandas as pd


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/predict", methods=["GET"])
@cross_origin()
def compute():
    params = {
        'type': request.args['type'],
        'data': [
            float(request.args['id']),
            float(request.args['Area']),
            float(request.args['MajorAxisLength']),
            float(request.args['MinorAxisLength']),
            float(request.args['Eccentricity']),
            float(request.args['ConvexArea']),
            float(request.args['EquivDiameter']),
            float(request.args['Extent']),
            float(request.args['Perimeter']),
            float(request.args['Roundness']),
            float(request.args['AspectRation'])
        ]
    }

    if(params['type'] == "SVM"):
        model = modelSVC
    elif(params['type'] == "logistic"):
        model = modelLogistic
    else:
        return

    [pred] = model.predict([params['data']])

    return jsonify({"predict": pred})


@app.route("/testSetPredictResults/<typeAlgorithm>", methods=["GET"])
@cross_origin()
def getTestSetPredictResults(typeAlgorithm):
    if(typeAlgorithm == 'SVM'):
        model = modelSVC
    elif(typeAlgorithm == 'logistic'):
        model = modelLogistic
    else:
        return

    data = pd.read_csv('btn3-testing.csv').values
    testX = data[:, :-1]
    testY = data[:, -1]

    predY = model.predict(testX)

    accuracy = str(accuracy_score(testY, predY))

    return jsonify({"predY": predY.tolist(), "testY": testY.tolist(), "accuracy": accuracy})


app.run()
