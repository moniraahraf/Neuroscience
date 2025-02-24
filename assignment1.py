import random
import math

def activation_function(x):
    return math.tanh(x)

def random_weight():
    return random.uniform(-0.5, 0.5)

w1 = random_weight()
w2 = random_weight()
b1 = 0.5
b2 = 0.7

while True:
    try:
        x = float(input("Enter input value: "))
        break 
    except ValueError:
        print("Invalid input! Please enter a numeric value.")

node1 = w1 * x + b1
af1 = activation_function(node1)

node2 = w2 * af1 + b2
af2 = activation_function(node2)

# Output
print("Output of the network:", af2)
