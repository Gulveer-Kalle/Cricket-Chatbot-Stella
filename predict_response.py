### File: data_preprocessing.py
import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

stemmer = PorterStemmer()
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
pattern_word_tags_list = []

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

def create_bot_corpus():
    global words, classes, pattern_word_tags_list

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            pattern_word_tags_list.append((pattern_words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words)
    stem_words = sorted(list(set(stem_words)))
    classes.sort()

    return stem_words, classes, pattern_word_tags_list

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for pattern_words, tag in pattern_word_tags_list:
        stem_pattern_words = get_stem_words(pattern_words, ignore_words)
        bag_of_words = [1 if word in stem_pattern_words else 0 for word in stem_words]
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for _, tag in pattern_word_tags_list:
        labels_encoding = [0] * len(classes)
        labels_encoding[classes.index(tag)] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus()
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    return train_x, train_y

if __name__ == "__main__":
    nltk.download('punkt')
    preprocess_train_data()


### File: train_bot.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_preprocessing import preprocess_train_data

# Train model
def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5')
    print("Model trained and saved as chatbot_model.h5")

if __name__ == "__main__":
    train_x, train_y = preprocess_train_data()
    train_bot_model(train_x, train_y)


### File: main.py
import nltk
import numpy as np
import pickle
import json
import random
import tensorflow as tf
from data_preprocessing import get_stem_words

nltk.download('punkt')

ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Load pre-trained data
model = tf.keras.models.load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def preprocess_user_input(user_input):
    input_words = nltk.word_tokenize(user_input)
    stemmed_words = get_stem_words(input_words, ignore_words)
    stemmed_words = sorted(set(stemmed_words))
    bag_of_words = [1 if word in stemmed_words else 0 for word in words]
    return np.array([bag_of_words])

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp, verbose=0)
    predicted_class_index = np.argmax(prediction[0])
    return predicted_class_index

def bot_response(user_input):
    predicted_class_index = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_index]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            return random.choice(intent["responses"])

if __name__ == "__main__":
    print("Hi, I am Stella! How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Stella: Goodbye!")
            break
        response = bot_response(user_input)
        print("Stella:", response)