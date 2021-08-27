from flask import Flask, jsonify, request
import pandas as pd
from joblib import load

app = Flask(__name__)

clf = load('calibrated_logistic_regression_classifier.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    basket = request.json['basket']
    zipCode = request.json['zipCode']
    totalAmount = request.json['totalAmount']
    p = probability(basket, zipCode, totalAmount)
    return jsonify({'probability': p}), 201

def probability(basket, zipCode, totalAmount):
    print("Processing request: {},{},{}".format(basket, zipCode, totalAmount))
    df_transformed_data = transform_data(basket, zipCode, totalAmount)
    prediction = predict(df_transformed_data, clf)
    return prediction

def transform_data(basket, zipCode, totalAmount):
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

if __name__ == "__main__":
	app.run(debug=True)


