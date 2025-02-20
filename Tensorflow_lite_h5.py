import tensorflow as tf
from keras.models import load_model
import os
# Convert the model
path1 = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved_model_14_at_d2/generator_500.h5"
path2 = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved_model_14_at_d2/TF"

# path = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved5/generator_300.h5"
model = load_model(path1)

# # Save it in the SavedModel format
# model.save(path2, save_format='tf')
# if not os.path.isfile(path2):
#     print("pass failed")
# else:
#     print("pass")
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved_model_14_at_d2/generator_500_3.tflite', 'wb') as f:
  f.write(tflite_model)