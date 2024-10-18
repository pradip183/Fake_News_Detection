from flask import Flask, request, jsonify, render_template
import fasttext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
import string

# Initialize the Flask app
app = Flask(__name__)

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
sequence_length = 8280  # Use the length that was used during training

# Preprocessing utilities
nltk_tokenizer = RegexpTokenizer(r'[A-Za-z]+')  # Regex tokenizer from NLTK
stemmer = SnowballStemmer("english")  # Snowball stemmer from NLTK

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
    
    # Tokenize and stem
    words = nltk_tokenizer.tokenize(text)
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Route for the homepage
@app.route('/')
def home():
    return render_template('success.html')

# Route for prediction
# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')

    # Preprocess the input text
    processed_text = wordopt(input_text)

    # Model 1: FastText prediction
    fasttext_prediction, fasttext_confidence = fasttext_model.predict(processed_text)
    fasttext_score = fasttext_confidence[0]  # Get the confidence score for the predicted label

    # Model 2: CNN prediction
    # Tokenize the processed text using the Keras tokenizer
    tokens = nltk_tokenizer.tokenize(processed_text)
    
    # Convert tokens to sequences using the Keras tokenizer's word_index
    sequences = [cnn_tokenizer.word_index.get(token) for token in tokens if token in cnn_tokenizer.word_index]
    
    # Check if sequences are empty
    if not sequences:
        return jsonify({"error": "No valid tokens found for prediction."}), 400

    # Pad sequences
    padded_sequences = pad_sequences([sequences], maxlen=sequence_length)

    cnn_prediction = cnn_model.predict(padded_sequences)[0][0]  # Get CNN model prediction

    # Combine predictions (example logic: you can adjust this)
    final_prediction = (fasttext_score + cnn_prediction) / 2  # Use fasttext_score instead of the label

    # Convert to label
    predicted_label = 1 if final_prediction >= 0.5 else 0

    # Define class mapping
    class_mapping = {0: "fake", 1: "true"}
    predicted_class_name = class_mapping[predicted_label]

    # Return the prediction result as JSON
    return jsonify({"prediction": predicted_class_name})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
