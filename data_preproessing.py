import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
movie_reviews=pd.read_csv("IMDB_Dataset[1].csv")
movie_reviews.isnull().values.any()
movie_reviews.shape
import seaborn as sns
sns.countplot(x="sentiment",data=movie_reviews)

import re

# 1. Define global variables and helper functions first to avoid NameError
TAG_RE = re.compile(r'<[^>]+>')

def remove_tag(text):
    return TAG_RE.sub('', text)

# 2. Main preprocessing function
def preprocess_text(sen):
    # Removing HTML tags
    sentence = remove_tag(sen)
    
    # Removing punctuations and numbers
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    
    # Single character removal
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence

# 3. Processing the dataset
review = []
sentences = list(movie_reviews['review']) # Renamed to 'sentences' (plural) to avoid confusion

for sen in sentences:
    review.append(preprocess_text(sen))

# Display the 5th review (Index 4) to verify
print(review[4])
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Convert sentiments (positive -> 1, negative -> 0)
converted = movie_reviews["sentiment"]
converted = np.array(list(map(lambda x: 1 if x == "positive" else 0, converted)))

# 2. Split the dataset into training and testing sets
review_train, review_test, converted_train, converted_test = train_test_split(
    review, converted, test_size=0.20, random_state=42
)
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize the Tokenizer keeping the top 5000 words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)

# Convert text sentences into sequences of numerical indices
review_train = tokenizer.texts_to_sequences(review_train)
review_test = tokenizer.texts_to_sequences(review_test)
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Define the maximum length of a sequence
maxlen = 100

# 2. Calculate the total vocabulary size (+1 to account for the padding token)
vocab_size = len(tokenizer.word_index) + 1

# 3. Pad sequences so they all have an identical length of 100
review_train = pad_sequences(review_train, padding='post', maxlen=maxlen)
review_test = pad_sequences(review_test, padding='post', maxlen=maxlen)
import numpy as np

embeddings_dictionary = dict()
# Open the downloaded GloVe file (Ensure the path matches where your file is located)
glove_file = open('/content/glove.6B.100d.txt', encoding='utf8')

for line in glove_file:
    records = line.split()
    word = records[0]
    # Convert vector dimensions to a float32 numpy array
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()
# Create an empty embedding matrix filled with zeros
embedding_matrix = np.zeros((vocab_size, 100))

# Map each unique word index to its GloVe vector representation
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector