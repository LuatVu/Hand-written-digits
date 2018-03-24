import sys
import os
import numpy as np

# sys.path.append('./src/')
sys.path.append('../fig/')


import mnist_loader
import network
import mnist
import gzip
import network2

import json
import random


temp1, temp2, image_tests = mnist_loader.load_data()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



# create model by network1 module
def create_model_1():
    net = network.Network([784, 30, 10])
    return net

# training model by network1 module
def training_model_1(net):
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# create model by network_2 module
def create_model_2():
    net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
    return net

# training model by network_2 module
def training_model_2(net):
    net.SGD(training_data, 30, 10, 0.5,
    lmbda= 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True)

def predict(net, image): #image is a numpy array 784x1 direction
    result = net.feedforward(image)
    result = np.argmax(result)
    return result

def plot_digit_image(image):
    #image argument is one image
    mnist.plot_mnist_digit(image)

def get_image(image_tests): # return a image set after flattened.
    flattened_images = image_tests[0]
    return [np.reshape(f, (-1, 28)) for f in flattened_images]


image_tests = get_image(image_tests)
datas = zip(image_tests, test_data)


def save_Model_1(net, filename):
    data = {
        "sizes": net.sizes,
        "weights": [w.tolist() for w in net.weights ],
        "biases": [b.tolist() for b in net.biases],        
    }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()

def save_Model_2(net, filename):
    net.save(filename)

def load_1(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = network.Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def load_2(filename):
    net = network2.load(filename)
    return net


# def main():
#     net = load_1("./data/Network1.bin")
#     for x, y in datas:
#         print "Predicted Result: ", np.argmax( net.feedforward(y[0]) )
#         plot_digit_image(x)        
#         try:
#             input ("Press enter to continue...")
#         except SyntaxError:
#             pass
#         os.system('clear')

# def main():
#     net = create_model_2()
#     training_model_2(net)
#     save_Model_2(net,"./data/Network2.bin")

def main():
    net = load_2("../data/Network2.bin")
    for x, y in datas:
        result = net.feedforward(y[0])
        print "Result: \n", result
        print "Digit: ", np.argmax( result )
        plot_digit_image(x)
        try:
            input("Press enter to continue...")
        except SyntaxError:
            pass
        os.system("clear")


if __name__ == "__main__":
    main()