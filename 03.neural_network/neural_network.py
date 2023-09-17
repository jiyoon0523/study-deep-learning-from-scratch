#CHAPTER3.신경망
#3.1. 퍼셉트론에서 신경망으로
##3.1.1 신경망의 예
##3.1.2 퍼셉트론 복습
##3.1.3 활성화 함수의 등장

#3.2 활성화 함수
##3.2.1 시그모이드 함수
##3.2.2 계단 함수 구현하기
import numpy as np
def step_function(x):
    # if x>0:
    #     return 1
    # else:
    #     return 0
    y= x>0
    return y.astype(np.int)

x= np.array([-1.0, 1.0, 2.0])
print(x)
y= x>0
print(y)
y= y.astype(int)
print(y)

##3.2.3 계단 함수의 그래프
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype= int)
x=np.arange(-5.0, 5.0, 0.1)
y= step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

##3.2.4 시그모이드 함수 구현하기
def sigmoid(x):
    return 1/ (1+np.exp(-x))
x=np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

t= np.array([1.0, 2.0, 3.0])
print(1.0+t)
print(1.0/t)

x=np.arange(-5.0, 5.0, 0.1)
y= sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

##3.2.5 시그모이드 함수와 계단 함수 비교
##3.2.6. 비선형 함수
def relu(x):
    return np.maximum(0,x)

#3.3 다차원 배열의 계산
##3.3.1 다차원 배열
B= np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

##3.3.2 행렬의 곱
A= np.array([[1,2], [3,4]])
print(A.shape)
B= np.array([[5,6],[7,8]])
print(B.shape)
print(np.dot(A,B))
    
A= np.array([[1,2,3], [4,5,6]])
print(A.shape) #2,3
B= np.array([[1,2], [3,4], [5,6]])
print(B.shape) #3,2
print(np.dot(A,B))

C= np.array([[1,2], [3,4]])
print(C.shape)
print(A.shape)
# print(np.dot(A,C)) #shapes not aligned error- dim1 != dim0

A= np.array([[1,2], [3,4], [5,6]])
print(A.shape)
B= np.array([7,8])
B.shape
print(B.shape)
print(np.dot(A,B))

##3.3.3 신경망에서의 행렬 곱
X=np.array([1,2])
print(X.shape)
W= np.array([[1,3,5], [2,4,6]])
print(W)
print(W.shape)
Y= np.dot(X,W)
print(Y)

#3.4 3층 신경망 구현하기
##3.4.1 표기법 설명
##3.4.2 각 층의 신호 전달 구현하기
X=np.array([1.0, 0.5])
W1= np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1= np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1= np.dot(X,W1)+B1
print(A1)

Z1= sigmoid(A1)
print(Z1)

W2= np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2= np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2= np.dot(Z1, W2)+B2
Z2= sigmoid(A2)

def identity_function(x):
    return x

W3= np.array([[0.1, 0.3], [0.2, 0.4]])
B3= np.array([0.1, 0.2])

A3= np.dot(Z2, W3)+ B3
Y= identity_function(A3) #or, Y=A3

##3.4.3. 구현 정리
def init_network():
    network= {}
    network['W1']= np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1']= np.array([0.1, 0.2, 0.3])
    network['W2']= np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2']= np.array([0.1, 0.2])
    network['W3']= np.array([[0.1, 0.3],[0.2, 0.4]])
    network['b3']= np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1= np.dot(x, W1)+b1
    z1= sigmoid(a1)
    a2= np.dot(z1, W2)+b2
    z2= sigmoid(a2)
    a3= np.dot(z2, W3)+b3
    y= identity_function(a3)

    return y

network= init_network()
x= np.array([1.0, 0.5])
y= forward(network, x)
print(y)


