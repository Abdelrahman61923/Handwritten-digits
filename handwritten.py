import os
import cv2
import numpy as np
import tensorflow as tf
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class HandwrittenDigitsRecognition:
    def __init__(self):
        self.model = None

    def train_model(self):
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_test, axis=1)

        model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=64, activation=tf.nn.sigmoid),
                tf.keras.layers.Dense(units=64, activation=tf.nn.tanh),
                tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=4)

        self.model = model

    def evaluate_model(self, X_test, y_test):
        val_loss, val_accuracy = self.model.evaluate(X_test, y_test)

        cm = confusion_matrix(y_test, np.argmax(self.model.predict(X_test), axis=-1))

        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Loss", val_loss])
        table.add_row(["Accuracy", val_accuracy])
        table.add_row(["Confusion Matrix", cm])

        print("Model evaluation results: ")
        print(table)

        return val_loss, val_accuracy

    def save_model(self, filename):
        self.model.save(filename)

    def load_saved_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def predict(self, image_paths):
        predictions = []
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)[:,:,0]
                img = np.invert(np.array([img]))
                prediction = self.model.predict(img)
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.title("Predicted Digit: {}".format(np.argmax(prediction)))
                plt.show()
                predictions.append(np.argmax(prediction))
            except:
                print("Error reading image:", image_path)
        return predictions

    def run(self):
        print("Handwritten Digits Recognition")

        train_new_model = True

        if train_new_model:
            self.train_model()
            mnist = tf.keras.datasets.mnist
            (_, _), (X_test, y_test) = mnist.load_data()
            self.evaluate_model(X_test, y_test)
            self.save_model('handwritten_digits.keras')
        else:
            self.load_saved_model('handwritten_digits.keras')

        images_dir = 'digits/'

        image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        predictions = self.predict(image_files)

        for image_path, prediction in zip(image_files, predictions):
            print("Image:", image_path, "Prediction:", prediction)

handwritten_recognition = HandwrittenDigitsRecognition()
handwritten_recognition.run()