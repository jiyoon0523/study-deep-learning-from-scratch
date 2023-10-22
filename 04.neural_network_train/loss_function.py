#04.신경망 학습
##4.1.데이터에서 학습한다
###4.1.1.데이터 주도 학습
###4.1.2.훈련 데이터와 시험 데이터

##4.2.손실 함수
###4.2.1.오차제곱합
import numpy as np

def sum_squares_error(y,t):
    return 0.5* np.sum((y-t)**2)

t= [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y= [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t))) #0.097

y= [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t))) #0.597

###4.2.2.교차 엔트로피 오차
def cross_entropy_error(y,t):
    delta= 1e-7
    return -np.sum(t*np.log(y+delta))

t= [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y= [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t))) #0.510

y= [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t))) #2.320

###4.2.3.미니배치 학습
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test)= load_mnist(normalize= True, one_hot_label= True)

print(x_train.shape)
print(t_train.shape)

train_size= x_train.shape[0]
batch_size= 10
batch_mask= np.random.choice(train_size, batch_size)
x_batch= x_train[batch_mask]
t_batch= t_train[batch_mask]

np.random.choice(60000, 10)

