import numpy as np
import matplotlib.pyplot as plt
import softsvm

def plot_combined_results(lambdas_small, small_results, lambdas_large, large_results, title):
    """
    Plot training and testing errors for both small and large sample sizes.
    Small sample errors are connected with lines, large sample errors are shown as points.
    
    Parameters:
        lambdas_small: List of λ values for small sample size.
        small_results: Dictionary containing train/test errors and min/max values for small samples.
        lambdas_large: List of λ values for large sample size.
        large_results: Dictionary containing train/test errors and min/max values for large samples.
        title: Title of the plot.
    """
    # Data for small sample size
    train_errors_small = small_results["train_errors"]
    test_errors_small = small_results["test_errors"]
    train_min_max_small = small_results["train_min_max"]
    test_min_max_small = small_results["test_min_max"]
    lambdas_log_small = np.log10(lambdas_small)

    train_min_small, train_max_small = zip(*train_min_max_small)
    test_min_small, test_max_small = zip(*test_min_max_small)

    # Data for large sample size
    train_errors_large = large_results["train_errors"]
    test_errors_large = large_results["test_errors"]
    lambdas_log_large = np.log10(lambdas_large)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Small sample size plot with error bars
    plt.errorbar(lambdas_log_small, train_errors_small,
                 yerr=[np.array(train_errors_small) - train_min_small, train_max_small - np.array(train_errors_small)],
                 label="Small Sample Train Error", fmt='-o', capsize=5, color='blue')
    plt.errorbar(lambdas_log_small, test_errors_small,
                 yerr=[np.array(test_errors_small) - test_min_small, test_max_small - np.array(test_errors_small)],
                 label="Small Sample Test Error", fmt='-o', capsize=5, color='orange')

    # Large sample size plot as points
    plt.scatter(lambdas_log_large, train_errors_large, label="Large Sample Train Error", color='green', s=80, marker='s')
    plt.scatter(lambdas_log_large, test_errors_large, label="Large Sample Test Error", color='red', s=80, marker='^')

    # Plot customization
    plt.xlabel(r'$\log_{10}(\lambda)$')
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_error(w, X, y):
    """
    Calculate the classification error.
    Parameters:
        w (numpy.ndarray): Linear predictor weights.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): True labels.
    Returns:
        float: Classification error rate.
    """
    predictions = np.sign(X @ w).flatten()
    errors = (predictions != y.flatten()).sum()
    return errors / len(y)

def run_experiment(trainX, trainy, testX, testy, lambdas, sample_size, repeats):
    """
    Run an SVM experiment with multiple λ values.
    Parameters:
        trainX, trainy: Training data and labels.
        testX, testy: Testing data and labels.
        lambdas: List of λ values to test.
        sample_size: Number of samples to randomly draw from the training data.
        repeats: Number of repetitions for small sample experiments.
    Returns:
        dict: Dictionary with results for each λ, including train and test errors.
    """
    results = {"train_errors": [], "test_errors": [], "train_min_max": [], "test_min_max": []}

    for l in lambdas:
        train_errors = []
        test_errors = []

        for _ in range(repeats):
            indices = np.random.choice(trainX.shape[0], sample_size, replace=False)
            sampleX = trainX[indices]
            sampley = trainy[indices]

            # Train soft-SVM
            w = softsvm.softsvm(l, sampleX, sampley)

            # Calculate train and test errors
            train_error = calculate_error(w, sampleX, sampley)
            test_error = calculate_error(w, testX, testy)

            train_errors.append(train_error)
            test_errors.append(test_error)

        # Store average errors and min/max values
        results["train_errors"].append(np.mean(train_errors))
        results["test_errors"].append(np.mean(test_errors))
        results["train_min_max"].append((min(train_errors), max(train_errors)))
        results["test_min_max"].append((min(test_errors), max(test_errors)))

    return results

def plot_results(lambdas, results, title):
    """
    Plot training and testing errors with error bars.
    Parameters:
        lambdas: List of λ values.
        results: Dictionary containing train/test errors and min/max values.
        title: Title of the plot.
    """
    train_errors = results["train_errors"]
    test_errors = results["test_errors"]
    train_min_max = results["train_min_max"]
    test_min_max = results["test_min_max"]

    lambdas_log = np.log10(lambdas)

    train_min, train_max = zip(*train_min_max)
    test_min, test_max = zip(*test_min_max)

    plt.figure(figsize=(10, 6))
    plt.errorbar(lambdas_log, train_errors, yerr=[np.array(train_errors) - train_min, train_max - np.array(train_errors)],
                 label="Train Error", fmt='-o', capsize=5)
    plt.errorbar(lambdas_log, test_errors, yerr=[np.array(test_errors) - test_min, test_max - np.array(test_errors)],
                 label="Test Error", fmt='-o', capsize=5)
    plt.xlabel(r'$\log_{10}(\lambda)$')
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load data
    data = np.load('EX2q2_mnist.npz')
    trainX, trainy = data['Xtrain'], data['Ytrain']
    testX, testy = data['Xtest'], data['Ytest']

    # Experiment 1: Small sample size
    lambdas_small = [10**n for n in range(-1, 12, 2)]
    small_results = run_experiment(trainX, trainy, testX, testy, lambdas_small, sample_size=100, repeats=10)
    #plot_results(lambdas_small, small_results, "Small Sample Size (100)")

    # Experiment 2: Large sample size 
    lambdas_large = [10**n for n in [1, 3, 5, 8]]
    large_results = run_experiment(trainX, trainy, testX, testy, lambdas_large, sample_size=1000, repeats=1)
    #plot_results(lambdas_large, large_results, "Large Sample Size (1000)") 
    plot_combined_results(lambdas_small, small_results, lambdas_large, large_results, "combined results")

if __name__ == "__main__":
    main()
