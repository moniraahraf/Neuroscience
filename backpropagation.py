import numpy as np 
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)

b1 = 0.5
b2 = 0.7


x = np.array([1.0]) 

# Forward pass
h = tanh(w1 * x + b1)
y = tanh(w2 * h + b2)
print("Output:", y)


target = np.array([0.8])  # Example target


lr = 0.1

# Backpropagation
error = target - y
d_y = error * tanh_derivative(y)

d_h = d_y * w2 * tanh_derivative(h)

w2 += lr * d_y * h
w1 += lr * d_h * x
b2 += lr * d_y
b1 += lr * d_h

print("Updated weights:", w1, w2)
print("Updated biases:", b1, b2)
