from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

import network2
import mnist_loader



def y_testdata_y_predict():
    net = network2.load("../data/Network2.bin")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    y_test = []
    y_predict = []

    for x, y in test_data:
        y = network2.vectorized_result(y).tolist()
        y_test.append(y)
        y_predict.append( network2.vectorized_result( np.argmax( net.feedforward(x) ) ).tolist() )
    return y_test, y_predict


def evaluation():
    y_test, y_predict = y_testdata_y_predict()

    precision = []
    recall = []
    f1 = []

    print "precision    recall    f1"
    for i in range(0,10):
        y = []
        y_pre = []

        for j in range(0, len(y_test) ):
            y.append( y_test[j][i])
            y_pre.append( y_predict[j][i])
        
        print precision_score(y, y_pre, average ='binary'), "  ",\
              recall_score(y, y_pre, average='binary'),"  ",\
              f1_score(y, y_pre, average='binary')


if __name__ == "__main__":
    evaluation()
    print "Done"