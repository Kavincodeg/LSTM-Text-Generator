import numpy as np
import tensorflow as tf
import string
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# -----------------------------
# TOKENIZATION
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# -----------------------------
# CREATE SEQUENCES (FIXED ✅)
# -----------------------------
token_list = tokenizer.texts_to_sequences([text])[0]

input_sequences = []
for i in range(1, len(token_list)):
    input_sequences.append(token_list[:i+1])

max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# -----------------------------
# LOAD OR TRAIN MODEL
# -----------------------------
if os.path.exists("model.h5"):
    print("Loading saved model...")
    model = load_model("model.h5")
else:
    print("Training new model...")

    model = Sequential([
        Embedding(total_words, 50, input_length=max_seq_len-1),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=300, verbose=1)

    model.save("model.h5")

# -----------------------------
# TOP-K SAMPLING
# -----------------------------
def sample_with_top_k(predictions, k=5, temperature=1.0):
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    top_k_indices = np.argsort(predictions)[-k:]
    top_k_probs = predictions[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)

    return np.random.choice(top_k_indices, p=top_k_probs)

# -----------------------------
# TEXT GENERATION
# -----------------------------
def generate_text(seed_text, next_words=20, temperature=0.9):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len-1,
            padding='pre'
        )

        predictions = model.predict(token_list, verbose=0)[0]

        predicted = sample_with_top_k(predictions, k=5, temperature=temperature)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # Avoid repetition
        if len(seed_text.split()) == 0 or seed_text.split()[-1] != output_word:
            seed_text += " " + output_word

    return seed_text

# -----------------------------
# USER INPUT
# -----------------------------
seed = input("Enter seed text: ")

print("\nGenerated Text:\n")
print(generate_text(seed))