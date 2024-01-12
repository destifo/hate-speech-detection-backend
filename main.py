import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import Flask, request


model = pickle.load(open("hsd-model.pk1", "rb"))
tokenizer = pickle.load(open("tokenizer.pk1", "rb"))
max_length = 55


def predict_sentiment(statement):
    # Tokenize the input statement
    tokenized_statement = tokenizer.texts_to_sequences([statement])

    padded_statement = pad_sequences(
        tokenized_statement, maxlen=max_length, padding="post"
    )

    predictions = model.predict(padded_statement)

    binary_predictions = np.round(predictions)

    return binary_predictions


def predict(statement):
    positive, negative = predict_sentiment(statement)[0]

    if positive > negative:
        return "Free"
    else:
        return "Hate"


def predict_sentiment(statement):
    # Tokenize the input statement
    tokenized_statement = tokenizer.texts_to_sequences([statement])

    padded_statement = pad_sequences(
        tokenized_statement, maxlen=max_length, padding="post"
    )

    predictions = model.predict(padded_statement)

    binary_predictions = np.round(predictions)

    return binary_predictions


def predict(statement):
    positive, negative = predict_sentiment(statement)[0]

    if positive > negative:
        return "Free"
    else:
        return "Hate"


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def handle_post_request():
    data = request.get_json()
    return {"message": "Prediction Completed", "verdict": predict(data["statement"])}


if __name__ == "__main__":
    app.run(port=5000)
