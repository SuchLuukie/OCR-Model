from random import random

# Selfmade simple dataset for testing the neural network

# Dataset sample
# [0, 0.5, 0.2]
# First int is whether is lethal or not (0 False, 1 True)
# Second float is 0-1
# Third float is 0-1
# Datapoint is lethal if second and third float combined is >= 1.3


class Dataset:
    def __init__(self):
        self.dataset = []
        self.size = 200
        
        self.generate_dataset()


    def generate_dataset(self):
        for _ in range(self.size):
            data = [0, random(), random()]
            
            if data[1] + data[2] >= 1.3:
                data[0] = 1

            self.dataset.append(data)