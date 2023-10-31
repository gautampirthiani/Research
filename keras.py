import numpy as np
import tensorflow as tf
import torch





# Define the Keras model
keras_model = tf.keras.Sequential()

# Define layers in Keras matching PyTorch model
keras_model.add(tf.keras.Conv2D(filters=2, kernel_size=(6, 1), strides=(2, 1), padding='valid', input_shape=(1, input_height, input_width)))
keras_model.add(tf.keras.ReLU())
keras_model.add(tf.keras.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True))

keras_model.add(tf.keras.Conv2D(filters=3, kernel_size=(5, 1), strides=(2, 1), padding='valid'))
keras_model.add(tf.keras.ReLU())
keras_model.add(tf.keras.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True))

keras_model.add(tf.keras.Conv2D(filters=5, kernel_size=(4, 1), strides=(2, 1), padding='valid'))
keras_model.add(tf.keras.ReLU())
keras_model.add(tf.keras.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True))

keras_model.add(tf.keras.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid'))
keras_model.add(tf.keras.ReLU())
keras_model.add(tf.keras.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True))

keras_model.add(tf.keras.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid'))
keras_model.add(tf.keras.ReLU())
keras_model.add(tf.keras.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True))

keras_model.add(tf.keras.Flatten())
keras_model.add(tf.keras.Dense(units=2))

# Load the PyTorch weights into the Keras model
pytorch_state_dict = torch.load('your_pytorch_model_weights.pth')
keras_layer_names = [layer.name for layer in keras_model.layers]

for layer_name in keras_layer_names:
    if layer_name in pytorch_state_dict:
        weights = pytorch_state_dict[layer_name].numpy()
        keras_model.get_layer(layer_name).set_weights([weights])

# Compile the Keras model
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Now, you can use keras_model for inference or further training in Keras.
