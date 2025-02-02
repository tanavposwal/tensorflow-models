import tensorflow as tf
import numpy as np

x = np.array([[200.0, 17.0], [120.0, 5.0], [425.0, 20.0], [212.0, 18.0]])

y = np.array([1, 0, 0, 1])

layer_1 = tf.keras.layers.Dense(3, activation="sigmoid")
layer_2 = tf.keras.layers.Dense(1, activation="sigmoid")

model = tf.keras.Sequential([layer_1, layer_2])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=200)

print(model.predict(x))
