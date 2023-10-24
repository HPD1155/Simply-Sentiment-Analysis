# %% [markdown]
# # This is a Basic Sentiment Analysis Project | @HPD1155
# # Libraries Used
# Tensorflow/Keras - Used for defining, building, and training the model
# Pandas - Used for loading the dataset
# Numpy - Used for working with arrays
# Matplotlib - Used for plotting the data
# Scikit-learn - Used for splitting the data into train and test sets
# Math - Used for rounding.

# %% [markdown]
# # Importing necessary libraries

# %%
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
import math

# %% [markdown]
# # Load in our dataset
# This will take out CSV filed named `data.csv` and create a dataframe out of it with **Pandas**.
# X is going to be the input data. It is getting the column named `text` from the dataframe.
# y is going to be the output data. It is getting the column named `label` from the dataframe.

# %%
df = pd.read_csv('data.csv') # Dataframe

X = df['text'] # Input data
y = df['label'] # Output data

# %% [markdown]
# # Preprocessing the data
# This section of code is going to do the following:
# - Tokenize the text (Eg. "Hello, my name is John" -> ['Hello', 'my', 'name', 'is', 'John'])
# - Convert the text to sequences (Eg. ['Hello', 'my', 'name', 'is', 'John'] -> [[3, 6, 8, 2, 4]])
# - pad the sequences so they are all the same dimension (Eg. [[3, 6, 8, 2, 4], [3, 6, 8, 2, 4]] -> [[0, 0, 0, 0, 0, 0, 3, 6, 8, 2, 4], [0, 0, 0, 0, 0, 0, 3, 6, 8, 2, 4]]) This is because the model expects all the sequences to be the same dimension or same length so each sequence has a use, there are no "holes" in the sequence.
# 
# # Important Note
# We aren't preprocessing the labels/y/output data because unlike X, the y is already in a numerical format that all has the same dimension. "1 and 0"

# %%
# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# Convert the text to sequences
X_seq = tokenizer.texts_to_sequences(X)

# pad the sequences so they are all the same dimension
X_pad = pad_sequences(X_seq)

# %% [markdown]
# We can now "visualize" what the X data might look like the the neural network. We can do this by printing out the first 5 sequences.

# %%
print("Preview of X_pad:", X_pad[:5])
print('Preview of X_seq: ', X_seq[:5])
# This is to show why we pad the sequences
print("Shape of X_seq 1 and X_seq 2:", len(X_seq[0]), len(X_seq[1]))
print("Shape of X_pad 1 and X_pad 2:", len(X_pad[0]), len(X_pad[1]))

# %% [markdown]
# Now we will split our data into train and test sets. This is so that we can train our model on the data, we can evaluate it's performance on unseen data.

# %%
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# %% [markdown]
# # Define the model
# We will now be defining the model with the following architecture and layers.
# In a nutshell, here is the job of each layer:
# - **Embedding Layer:** This layer takes in an integer matrix of size (input_dim, output_dim) as input and produces an output matrix of size (input_dim, output_dim) as output. This layer is used to learn word vectors.
# - **GlobalAveragePooling1D:** This layer takes in a list of vectors and returns a vector with the average of the list of vectors.
# - **Dropout:** This layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
# - **Dense:** This layer has 32 units which are used to compute an output and uses the relu activation function.
# - **Dense:** This layer has 16 units which are used to compute an output and uses the relu activation function.
# - **Dense:** This layer has 1 unit which is used to compute an output and uses the sigmoid activation function to output a value between 0 and 1 or the probability of the input being true/positive.

# %%
model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# %% [markdown]
# At this point, we will now compile the model. This means that we will specify the loss function, the optimizer, and the metrics we are interested in evaluating the model.
# What we will use:
# - **Loss:** Binary-Crossentropy
# - **Optimizer:** Adam
# - **Metrics:** Accuracy

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [markdown]
# Now we will train the model on our data. `X_train` and `y_train`
# We will be using 10 epochs to train the model which means that we will be training the model for 10 iterations with a default batch size of 32.

# %%
history = model.fit(X_train, y_train, epochs=10)

# %% [markdown]
# Now we can evalute how well the model performs on unseen data.

# %%
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# %% [markdown]
# As you can see, our accuracy is 84%, but it was 95% when training. This might be caused by overfitting. You can add a validation set to avoid this.
# Now lets graph the loss and accuracy of our model when it was training

# %%
# This will be our training loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# %%
# This will be our accuracy
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

# %% [markdown]
# As you can see, as we trained the model, the loss decreased and the accuracy increased. This is a good sign.
# Lets now try getting the AI to predict whether a sentence is positive or negative from a user input.

# %%
# Predict on test data
prediction = model.predict(X_test)
print(prediction[:5])

# New data
raw_input = input("Enter a sentence: ")
print('User sentence:', raw_input)
encoded_input = tokenizer.texts_to_sequences([raw_input])
padded_input = pad_sequences(encoded_input)

prediction = model.predict(padded_input)
print("The probability of the sentence being positive is: ", "{:.{precision}f}".format(float(prediction[0][0]), precision=4))

# %% [markdown]
# That is it for this project. Thank you guys!!


