import json
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../src/")
import mnist_loader
import network2



def run_network():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    print "Training the network using Quadratic cost function"
    net = network2.Network( [784, 30, 10], cost = network2.QuadraticCost)    

    vc_1, va_1, tc_1, ta_1  \
          = net.SGD(training_data, 30, 10, 0.025, lmbda=5.0,
                    evaluation_data=validation_data,
                    monitor_evaluation_accuracy=True)
    
    print "Training the network using Cross-Entropy cost function"
    net = network2.Network( [784, 30, 10], cost = network2.CrossEntropyCost)
    vc_2, va_2, tc_2, ta_2  \
          = net.SGD(training_data, 30, 10, 0.025, lmbda=5.0,
                    evaluation_data=validation_data,
                    monitor_evaluation_accuracy=True)

    f = open("./evaluation_other_costfunction.json","w")
    json.dump({
        "Quadratic_Cost":[vc_1, va_1, tc_1, ta_1],
        "Cross-Entropy_Cost":[vc_2, va_2, tc_2, ta_2]
    },f)
    f.close()

def make_plot():
    f = open("./evaluation_other_costfunction.json","r")
    result = json.load(f)
    f.close()
    vc_1, va_1, tc_1, ta_1 = result["Quadratic_Cost"]
    vc_2, va_2, tc_2, ta_2 = result["Cross-Entropy_Cost"]

    va_1 = [x/100.0 for x in va_1]
    va_2 = [x/100.0 for x in va_2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot( np.arange(0,30, 1),  va_1, color='#2A6EA6',
           label="Quadratic_Cost")
    ax.plot( np.arange(0, 30, 1), va_2, color='#FFA933',
           label="Cross-Entropy_Cost")
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([ 85, 100])
    ax.set_title('Classification (eta = 0.025)')
    plt.legend(loc="lower right")
    plt.show()

