from random import random

# Selfmade simple dataset for testing the neural network

# Dataset sample
# [1, 4.2, 5.2]
# First int is whether is lethal or not (0 False, 1 True)
# Second float is 0-1, no impact on lethal or not
# Third float is 0-1, >= 0.7 is lethal

class Dataset:
    def __init__(self):
        self.dataset = []
        self.size = 200


    def generate_dataset(self):
        for _ in range(self.size):
            data = [0, random(), random()]
            if data[2] >= 0.7:
                data[0] = 1
            
            self.dataset.append(data)