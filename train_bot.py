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