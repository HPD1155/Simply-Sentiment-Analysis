# Basic Sentiment Analysis Project | Aadi Kulkarni

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data.csv')

X = df['text']
y = df['label']

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# Convert the text to sequences
X_seq = tokenizer.texts_to_sequences(X)

# pad the sequences so they are all the same dimension
X_pad = pad_sequences(X_seq)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)


'''
We don't need to preprocess the y data as it is already numerical and in the format we want it to be. :)
'''

# Define the model

model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))