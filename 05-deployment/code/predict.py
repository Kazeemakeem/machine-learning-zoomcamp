import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn) # turns numpy boolean to python boolean.
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# you can run this app in the terminal with gunicorn to avoid WSGI server warning
#  "gunicorn --bind 0.0.0.0:9696 predict:app" - lines 30-31 won't run and as a result are irrelevant with this command. You can also use waitress with "waitress-serve --listen=0.0.0.0:9696 predict:app"

# Pipenv
#     install - pip install pipenv
#     install all your dependencies in current venv with pipenv install
#     run pipenv install to reinstall dependencies
#     run pipenv shell to run your package using the current venv