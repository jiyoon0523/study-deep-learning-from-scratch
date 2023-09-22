##3.5.4 출력층의 뉴런 수 정하기

#3.6 손글씨 숫자 인식
##3.6.1 MNIST 데이터셋

import sys, os
import numpy as np
import tensorflow as tf

# sys.path.append(os.pardir)
# sys.path.append(r"D:\Projects\study-deep-learning-from-scratch\03.neural_network")
# from dataset.mnist import load_mnist
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten= True, normalize= False)

(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

from PIL import Image

def img_show(img):
    pil_img= Image.fromarray(np.uint8(img))
    pil_img.show()

img= x_train[0]
label= t_train[0]
print(label)

# print(img.shape)
#img= img.reshape(784, 1) #flatten be like this
# print(img.shape)

img_show(img)
