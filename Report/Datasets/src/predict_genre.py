import sys
import numpy as np

from Datasets.src.Preprocessor import Preprocessor
from Datasets.src.OneHotEncoder import OneHotEncoder
from Datasets.src.Vectorizer import Vectorizer
from Datasets.src.Model import Model

from keras.models import model_from_json

def load_keras_model(model_path='Datasets/model/'):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model

def main(subtitle_text):

    p = Preprocessor()      # Preprocessor is initialized
    v = Vectorizer()        # Vectorizer is initialized
    v.load()                # Vectorizer is loaded
    enc = OneHotEncoder()   # Encoder is initialized
    enc.load()              # Encoder is loaded

    # READ INPUT SUBTITLE AND PREPROCESS
    subtitle_text_processed = p.preprocess_document(subtitle_text)

    # VECTORIZE PREPROCESSED SUBTITLE
    X   = v._transform(subtitle_text_processed, method='count')
    X   = np.reshape(X, (1, -1))

    model       = load_keras_model()        # Load model
    pred_onehot = model.predict(X)          # Make prediction
    genre       = enc.decode(pred_onehot)   # Encode one-hot label

    return genre