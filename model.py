from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Initialize the Sequential model
model = Sequential()

# Create and add the Pre-trained Embedding Layer using your GloVe matrix
embedding_layer = Embedding(
    vocab_size, 
    100, 
    weights=[embedding_matrix], 
    input_length=maxlen, 
    trainable=False
)
model.add(embedding_layer)

# Add an LSTM layer with 128 units to capture sequential context
model.add(LSTM(128))

# Add the final Dense output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Display the complete architecture summary
model.summary()

# Compile the model with optimization settings
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['acc']
)
# Train the model and log progress into the history variable
history = model.fit(
    review_train, 
    converted_train, 
    batch_size=128, 
    epochs=6,               # Set to 6 epochs as shown in your final screenshot
    verbose=1, 
    validation_split=0.2    # Reserves 20% of training data for validation
)

# Evaluate the final model performance on the independent test dataset
score = model.evaluate(review_test, converted_test, verbose=1)

# Print the final accuracy and loss results cleanly
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
# Convert the custom string instance into sequence indices
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

# Pad the flattened list to the same maxlen sequence size
instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

# Generate the probability score (Close to 1 = Positive, Close to 0 = Negative)
model.predict(instance)
import matplotlib.pyplot as plt

# --- 1. Plot Model Accuracy ---
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# --- 2. Plot Model Loss ---
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import pickle

# 1. Save the Tokenizer object
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 2. Save the trained Keras LSTM model
model.save('sentiment_lstm_model.h5')
print("Model and Tokenizer exported successfully!")