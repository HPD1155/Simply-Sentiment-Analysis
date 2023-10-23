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

# Load the dataset
df = pd.read_csv('data.csv')

X = df['text']
y = df['label']