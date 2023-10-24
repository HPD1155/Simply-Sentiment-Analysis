import tensorflow.keras.models as models
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# load the dataframe so tokenizer can fit on it
df = pd.read_csv('data.csv')

# Set the text column as the input data
X = df['text']

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# load the model
model = models.load_model('model.keras')

# Get the user input to run from the saved model
raw_input = input("Enter a sentence: ")

# Convert the user input to a sequence
encoded_input = tokenizer.texts_to_sequences([raw_input])

# Pad the sequence
padded_input = pad_sequences(encoded_input)

# Run the model
prediction = model.predict(padded_input)

# Print the prediction
print("The probability of the sentence being positive is: ", "{:.{precision}f}".format(float(prediction[0][0]), precision=4))