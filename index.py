
from flask import Flask, request, jsonify, send_file, render_template
import os
import jwt
import time
import secrets
import fasttext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from web3 import Web3
from eth_account.messages import encode_defunct
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__)

# Secret key for JWT encoding/decoding
secret_key = 'mySecretKey'

# Load fastText model
fasttext_model = fasttext.load_model('news_model.bin')

# Load CNN model
cnn_model = load_model('fasttext-cnn-model.h5')

# Load the tokenizer used for CNN (from Keras)
with open('tokenizer.pickle', 'rb') as handle:
    cnn_tokenizer = pickle.load(handle)

# Load the embedding matrix
embedding_matrix = np.load('embedding_matrix.npy')

# Define the sequence length (same as during CNN model training)
sequence_length = 8280

# Preprocessing utilities
nltk_tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w', '', text)
    words = nltk_tokenizer.tokenize(text)
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Static folder setup
@app.route('/')
def index():
    return send_file(os.path.join(os.getcwd(), 'index.html'))

# GET route to retrieve a nonce value for signing
@app.route('/api/nonce', methods=['GET'])
def get_nonce():
    nonce = secrets.token_hex(32)
    return jsonify({'nonce': nonce})

# POST route to handle login and token creation
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    signed_message = data.get('signedMessage')
    message = data.get('message')
    address = data.get('address')
    
    try:
        w3 = Web3()
        message_hash = encode_defunct(text=message)
        recovered_address = w3.eth.account.recover_message(message_hash, signature=signed_message)
        
        if recovered_address.lower() != address.lower():
            return jsonify({'error': 'Invalid signature'}), 401
        
        token = jwt.encode({'address': address, 'exp': time.time() + 600}, secret_key, algorithm='HS256')
        return jsonify(token=token)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 401

# Endpoint for verifying the JWT token
@app.route('/verify', methods=['POST'])
def verify():
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Invalid token'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        if decoded.get('exp') < time.time():
            return jsonify('tokenExpired')
        return jsonify('ok')
    
    except jwt.ExpiredSignatureError:
        return jsonify('tokenExpired'), 401
    
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

# Serve the success page
@app.route('/success', methods=['GET'])
def success():
    return send_file(os.path.join(os.getcwd(), 'success.html'))

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')

    processed_text = wordopt(input_text)

    fasttext_prediction, fasttext_confidence = fasttext_model.predict(processed_text)
    fasttext_score = fasttext_confidence[0]

    tokens = nltk_tokenizer.tokenize(processed_text)
    sequences = [cnn_tokenizer.word_index.get(token) for token in tokens if token in cnn_tokenizer.word_index]

    if not sequences:
        return jsonify({"error": "No valid tokens found for prediction."}), 400

    padded_sequences = pad_sequences([sequences], maxlen=sequence_length)
    cnn_prediction = cnn_model.predict(padded_sequences)[0][0]

    final_prediction = (fasttext_score + cnn_prediction) / 2
    predicted_label = 1 if final_prediction >= 0.5 else 0

    class_mapping = {0: "fake", 1: "true"}
    predicted_class_name = class_mapping[predicted_label]

    return jsonify({"prediction": predicted_class_name})

if __name__ == '__main__':
    app.run(port=3000)