##3.5.4 출력층의 뉴런 수 정하기

#3.6 손글씨 숫자 인식
##3.6.1 MNIST 데이터셋

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =\
load_mnist(flatten= True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
