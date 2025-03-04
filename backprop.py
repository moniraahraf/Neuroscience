import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

i1, i2 = 0.05, 0.10

w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45
w7, w8 = 0.50, 0.55

b1, b2 = 0.35, 0.60

h1_input = (i1 * w1) + (i2 * w2) + b1
h2_input = (i1 * w3) + (i2 * w4) + b1

h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

o1_input = (h1_output * w5) + (h2_output * w6) + b2
o2_input = (h1_output * w7) + (h2_output * w8) + b2

o1_output = sigmoid(o1_input)
o2_output = sigmoid(o2_input)

expected_o1, expected_o2 = 0.01, 0.99

error_o1 = 0.5 * (expected_o1 - o1_output) ** 2
error_o2 = 0.5 * (expected_o2 - o2_output) ** 2

learning_rate = 0.5

delta_o1 = (o1_output - expected_o1) * sigmoid_derivative(o1_output)
delta_o2 = (o2_output - expected_o2) * sigmoid_derivative(o2_output)

w5 -= learning_rate * delta_o1 * h1_output
w6 -= learning_rate * delta_o1 * h2_output
w7 -= learning_rate * delta_o2 * h1_output
w8 -= learning_rate * delta_o2 * h2_output

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(h1_output)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(h2_output)

w1 -= learning_rate * delta_h1 * i1
w2 -= learning_rate * delta_h1 * i2
w3 -= learning_rate * delta_h2 * i1
w4 -= learning_rate * delta_h2 * i2

print(f"Output O1: {o1_output}")
print(f"Output O2: {o2_output}")
print(f"Error O1: {error_o1}")
print(f"Error O2: {error_o2}")
print(f"Updated Weights:\n w1={w1}, w2={w2}, w3={w3}, w4={w4},\n w5={w5}, w6={w6}, w7={w7}, w8={w8}")
