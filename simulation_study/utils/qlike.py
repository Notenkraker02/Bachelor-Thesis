import numpy as np

def get_qlike(predictions, Y_test):
    Y_test = Y_test.squeeze()
    qlike = np.mean((Y_test/predictions) - np.log(Y_test/predictions) - 1)
    return qlike