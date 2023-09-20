#3.5 출력층 설계하기

##3.5.1 항등 함수와 소프트맥스 함수 구현하기
import numpy as np

a= np.array([0.3, 2.9, 4.0])
exp_a= np.exp(a)
print(exp_a)

sum_exp_a= np.sum(exp_a)
print(sum_exp_a)

y= exp_a/ sum_exp_a
print(y)

def softmax(a):
    exp_a= np.exp(a)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/ sum_exp_a

    return y

print(softmax(a))

##3.5.2 소프트맥스 함수 구현 시 주의점
a= np.array([1010, 1000, 990])
print(softmax(a)) #overflow in exp #[nan nan nan]

c= np.max(a)
print(a-c)
print(softmax(a-c))

def better_softmax(a):
    c= np.max(a)
    exp_a= np.exp(a-c)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/ sum_exp_a
    
    return y

print(better_softmax(a))

#3.5.3 소프트맥스 함수의 특징

