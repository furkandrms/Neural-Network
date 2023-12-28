import numpy as np
from layer import Layer
from activation import Activation


class ReLu: 
    @staticmethod 
    def activate(x): 
        return np.maximum(0,x)


    @staticmethod
    def prime(x): 
        return np.where(x > 0, 1, 0)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
"""class T(Activation):
    def __init__(self):
        def t(n):
            return 1 / (n**2 + n - 1)

        def t_prime(n):
            # Buraya t fonksiyonunun türevini hesaplayan kodu ekleyin
            pass

        super().__init__(t, t_prime)


class Fib(Activation):
    def __init__(self):
        super().__init__()

    def fib(self, x):
        return np.sqrt(5) / ((1 + np.sqrt(5) / 2)**x - (1 - np.sqrt(5) / 2)**x)"""

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)


class T(Activation):
    def __init__(self):
        def t(n):
            return 1 / (n**2 + n - 1)

        def t_prime(n):
            # The derivative of the t function
            return -(2*n + 1) / (n**2 + n - 1)**2

        super().__init__(t, t_prime)


class Fib(Activation):
    def __init__(self):
        def fib(n):
            return np.sqrt(5) / ((1 + np.sqrt(5) / 2)**np.floor(n) - (1 - np.sqrt(5) / 2)**np.floor(n))

        
    def __init__(self):
        def fib(n):
            return np.sqrt(5) / ((1 + np.sqrt(5) / 2)**np.floor(n) - (1 - np.sqrt(5) / 2)**np.floor(n))

        def dummy_derivative(n):
            # Dummy derivative function as a placeholder
            return np.zeros_like(n)

        super(Fib, self).__init__(fib, dummy_derivative)
    
