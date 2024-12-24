import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    import numpy as np
from cvxopt import matrix, solvers

def softsvm(l, trainX, trainy):
    """
    Implements the soft-SVM algorithm.
    
    Parameters:
        l (float): The parameter λ of the soft SVM algorithm.
        trainX (numpy.ndarray): A 2-D matrix of size m × d, where m is the sample size and d is the dimension of the examples.
        trainy (numpy.ndarray): A column vector of length m with labels yi ∈ {−1, 1}.
        
    Returns:
        numpy.ndarray: The linear predictor w, a column vector in R^d.
    """
    # Get the number of samples (m) and the dimension of examples (d)
    m, d = trainX.shape

    # Convert trainy to a diagonal matrix for constraints
    y_diag = np.diag(trainy.flatten())
    
    # Construct the matrices for the quadratic programming problem
    H = np.block([
        [np.eye(d), np.zeros((d, m))],
        [np.zeros((m, d)), np.zeros((m, m))]
    ]) * (2 * l)  # Regularization term
    H = matrix(H) #maybe need to add a small epsilon if there is an error

    u = np.concatenate([np.zeros(d), np.ones(m) / m])
    u = matrix(u)

    # Construct A for the constraints
    upper_block = np.hstack([y_diag @ trainX, np.eye(m)])  # y_diag * trainX and slack variables
    lower_block = np.hstack([np.zeros((m, d)), np.eye(m)])  # 0 matrix and eye(m)
    A = np.vstack([upper_block, lower_block])  # Combine both blocks
    A = matrix(A)

    # Construct v for the constraints
    v = np.concatenate([np.ones(m), np.zeros(m)])  # [1...1] for upper block and [0...0] for lower block
    v = matrix(v)

    # Solve the quadratic programming problem
    sol = solvers.qp(H, u, -A, -v)

    # Extract w (linear predictor) from the solution
    solution = np.array(sol["x"])
    w = solution[:d]  # First d elements correspond to w

    return w


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
