import numpy as np

def get_qlike(predictions, Y_test):
    qlikes = {}
    for model, pred in predictions.items():
        Y_test = Y_test.squeeze()
        qlike = np.mean((Y_test/pred) - np.log(Y_test/pred) - 1)
        qlikes[model] = qlike
    return qlikes