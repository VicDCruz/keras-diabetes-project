from numpy import loadtxt
import tensorflow as tf


def load_dataset():
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    return dataset[:, 0:8], dataset[:, 8]


def build_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(24, input_shape=(8,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def predict(model: tf.keras.models.Sequential, X):
    return (model.predict(X) > 0.5).astype(int)


if __name__ == '__main__':
    print('Hello, world')
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X, y = load_dataset()
    model.fit(X, y, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

    predictions = predict(model, X)
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
