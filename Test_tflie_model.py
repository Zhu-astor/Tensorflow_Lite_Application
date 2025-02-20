import tensorflow as tf
import numpy as np
interpreter = tf.lite.Interpreter(model_path="C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved_model_14_at_d2/generator_500_2.tflite")
interpreter.allocate_tensors()

# 获取输入输出张量的信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备输入数据
input_data = np.random.rand(*input_details[0]['shape']).astype(np.float32)

# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 获取输出张量
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
print(output_data)