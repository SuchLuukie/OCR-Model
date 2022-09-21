# OCR Model
The goal of this project is to create a neural network from scratch and teaching it OCR (Optical Character Recognition)


# Artificial Neural Network (ANN)
* Basic prototype - Using what I found in my research I'm developing a small ANN that uses weights and biases to process and a small self made dataset. (See below.) In this small network you can manually adjust the weights and biases to see the neural network adjust it's output.



# Basic dataset
Each datapoint consists of 1 int and 2 floats ([0, 0.0, 0.0]).
The two floats determine whether the datapoint is true or false, when the floats are combined and are >= some threshold, the first integer changes if it's true or false.


# Visualisation
I display the dataset in a graph in matplotlib, with a colour map behind it.
The colour map runs every pixel through the ANN and determines what colour the pixel should be from that output.

* Basic prototype visualisation:
    <img src="/resources/prototype.png" width="360" height="240"/>