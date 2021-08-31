from flask import Flask, jsonify, request
import pandas as pd
from joblib import load

app = Flask(__name__)

clf = load('calibrated_logistic_regression_classifier.joblib')
imputed_values = load('imputed_values.joblib')
imputed_zipcode = imputed_values['zipcode']
imputed_totalAmount = imputed_values['totalAmount']

basket_failures = []
zipCode_failures = []
totalAmount_failures = []
MIN_NUMBER_OF_REQUESTS = 10

@app.route('/predict', methods=['POST'])
def predict():
    try:
        basket = request.json['basket']
        zipCode = request.json['zipCode']
        totalAmount = request.json['totalAmount']
        p = probability(basket, zipCode, totalAmount)
        return jsonify({'probability': p}), 201
    except:
        return "There was an error"

def update_failures_list(update_value, failure_list):
    if len(failure_list) >= MIN_NUMBER_OF_REQUESTS:
        del failure_list[0]
        failure_list.append(update_value)
    else:
        failure_list.append(update_value)

def probability(basket, zipCode, totalAmount):
    print("Processing request: {},{},{}".format(basket, zipCode, totalAmount))
    df_transformed_data = transform_data(basket, zipCode, totalAmount)
    prediction = predict(df_transformed_data, clf)
    return prediction

def transform_data(basket, zipCode, totalAmount):
    if not basket:
        update_failures_list(1, basket_failures)
        basket = []
    else:
        update_failures_list(0, basket_failures)

    if not zipCode:
        update_failures_list(1, zipCode_failures)
        zipCode = imputed_zipcode
    else:
        update_failures_list(0, zipCode_failures)

    if not totalAmount:
        update_failures_list(1, totalAmount_failures)
        totalAmount = imputed_totalAmount
    else:
        update_failures_list(0, totalAmount_failures)

    df = pd.DataFrame({"basket": [basket], "zipCode": [zipCode], "totalAmount": [totalAmount]})

    # basket
    for i in range(0, 6):
        df[f"c_{i}"] = df["basket"].map(lambda x: x.count(i))
    df = df.drop("basket", axis=1)

    # zipCode
    df["zipCode"] = pd.Categorical(df["zipCode"], categories=list(range(100, 1001)))
    dummies = pd.get_dummies(df.zipCode)
    df = pd.concat([df, dummies], axis=1).drop("zipCode", axis=1)

    return df 

def predict(df, clf):
    return format(clf.predict_proba(df)[0][1], '.8f')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    number_of_requests = len(basket_failures)

    if number_of_requests < MIN_NUMBER_OF_REQUESTS:
        return f"There have not been over {MIN_NUMBER_OF_REQUESTS} requests to this server yet"
    else:
        return jsonify({
            'basket failures': sum(basket_failures),
            'zip code failures': sum(zipCode_failures),
            'total amount failures': sum(totalAmount_failures)
            }), 201

if __name__ == "__main__":
	app.run(debug=True)


