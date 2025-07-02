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