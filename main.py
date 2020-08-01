import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set up the training data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
for i, c in enumerate(celsius):
    print("{} degree Celsuis = {} degree Fahrenheit".format(c, fahrenheit[i]))

# Create the model
layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

# Show
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

#
predict_data = [100, 200]
print(model.predict([100, 200]))

#
def get_fahrenheit(c):
    return c*1.8+32.0

ret_celsuis = []
for data in predict_data:
    ret_celsuis.append(get_fahrenheit(data))

print(ret_celsuis)

#
print("These are the layer variables: {}".format(layer0.get_weights()))