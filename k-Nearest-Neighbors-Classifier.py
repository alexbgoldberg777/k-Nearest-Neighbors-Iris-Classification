import pandas as pd
import math
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np

# Read in data
data = pd.read_csv('iris.csv')

# Normalize the data so all attributes are between 0 and 1
normalized_col1 = []
normalized_col2 = []
normalized_col3 = []
normalized_col4 = []
for val in data['Sepal-Length']:
    normalized_col1.append((val - data.min()[0]) / (data.max()[0] - data.min()[0]))
for val in data['Sepal-Width']:
    normalized_col2.append((val - data.min()[1]) / (data.max()[1] - data.min()[1]))
for val in data['Petal-Length']:
    normalized_col3.append((val - data.min()[2]) / (data.max()[2] - data.min()[2]))
for val in data['Petal-Width']:
    normalized_col4.append((val - data.min()[3]) / (data.max()[3] - data.min()[3]))
normalized_data = pd.DataFrame(data={
    'Sepal-Length' : normalized_col1,
    'Sepal-Width' : normalized_col2,
    'Petal-Length' : normalized_col3,
    'Petal-Width' : normalized_col4,
    'Class' : data['Class']
})

# Shuffles the rows of the normalized data and returns a tuple with a training set and a test set in that order
def train_test_split():
    data_sample = normalized_data.sample(frac=1)
    return (data_sample.iloc[:120], data_sample.iloc[120:])

# Uses k-Nearest-Neighbors to make a prediction on some set of values to test based on a set of training values
def make_predictions(training_set, test_set, k):

    predicted_classes = [] # Stores the final predictions of each instance as a tuple containing the indices and the predicted classes

    for instance in test_set.iterrows():
        nearest_neighbors = [] # Stores a tuple containing the indices of the k-nearest neighbors, their Euclidean distances, and their classes
        num_setosa = 0
        num_versicolor = 0
        num_virginica = 0

        # Finds the k instances that are closest in the sum of the Euclidean distances for each attribute to the current instance
        for neighbor in training_set.iterrows():
            Sepal_Length_distance = math.sqrt( (instance[1][0] - neighbor[1][0]) ** 2 )
            Sepal_Width_distance = math.sqrt( (instance[1][1] - neighbor[1][1]) ** 2 )
            Petal_Length_distance = math.sqrt( (instance[1][2] - neighbor[1][2]) ** 2 )
            Petal_Width_distance = math.sqrt( (instance[1][3] - neighbor[1][3]) ** 2 )
            Euclidean_distance = Sepal_Length_distance + Sepal_Width_distance + Petal_Length_distance + Petal_Width_distance
            if len(nearest_neighbors) < k:
                nearest_neighbors.append((neighbor[0], Euclidean_distance, neighbor[1][4]))
            else:
                max_nearest_neighbor = max(nearest_neighbors, key=itemgetter(1))
                if Euclidean_distance < max_nearest_neighbor[1]:
                    nearest_neighbors[nearest_neighbors.index(max_nearest_neighbor)] = ((neighbor[0], Euclidean_distance, neighbor[1][4]))

        for neighbor in nearest_neighbors:
            if neighbor[2] == 'Iris-setosa':
                num_setosa += 1
            elif neighbor[2] == 'Iris-versicolor':
                num_versicolor += 1
            else:
                num_virginica += 1

        # Classifies the current instance as the label of the majority of its k neighbors, with ties broken at random
        if num_setosa > num_versicolor and num_setosa > num_virginica:
            predicted_classes.append((instance[0], 'Iris-setosa'))
        elif num_versicolor > num_setosa and num_versicolor > num_virginica:
            predicted_classes.append((instance[0], 'Iris-versicolor'))
        elif num_virginica > num_setosa and num_virginica > num_versicolor:
            predicted_classes.append((instance[0], 'Iris-virginica'))
        elif num_setosa == num_versicolor and num_setosa == num_virginica:
            random_tiebreaker = random.randint(1, 3)
            if random_tiebreaker == 1:
                predicted_classes.append((instance[0], 'Iris-setosa'))
            elif random_tiebreaker == 2:
                predicted_classes.append((instance[0], 'Iris-versicolor'))
            else:
                predicted_classes.append((instance[0], 'Iris-virginica'))
        elif num_setosa == num_versicolor:
            random_tiebreaker = random.randint(1, 2)
            if random_tiebreaker == 1:
                predicted_classes.append((instance[0], 'Iris-setosa'))
            else:
                predicted_classes.append((instance[0], 'Iris-versicolor'))
        elif num_setosa == num_virginica:
            random_tiebreaker = random.randint(1, 2)
            if random_tiebreaker == 1:
                predicted_classes.append((instance[0], 'Iris-setosa'))
            else:
                predicted_classes.append((instance[0], 'Iris-virginica'))
        else:
            random_tiebreaker = random.randint(1, 2)
            if random_tiebreaker == 1:
                predicted_classes.append((instance[0], 'Iris-versicolor'))
            else:
                predicted_classes.append((instance[0], 'Iris-virginica'))

    assert(len(test_set.index) == len(predicted_classes)) # Check that all rows in the test_set were accounted for

    return predicted_classes

# Computes the accuracy of the k-NN algorithm with different values of k by calculating the percentage of correct predictions
def compute_accuracy(training_set, test_set):
    k_vals = [num for num in range(1, 52, 2)] # List of k-values to test

    number_correct = 0 # Number of correct predictions
    accuracies = [] # List of accuracies of each run as a tuple containing the k-value used and the accuracy of that run

    for k in k_vals:
        current_run = make_predictions(training_set, test_set, k)
        for i in range(len(current_run)):
            if test_set.iloc[i][4] == current_run[i][1]:
                number_correct += 1
        accuracies.append((k, number_correct/len(current_run)))
        number_correct = 0

    return accuracies

# Splits the normalized data into training and testing data 20 times, and for each of these 20 times, runs the k-NN algorithm and computes
# accuracies for different values of k. Both sets are used as the test set each time, with the training set being used for training each time.
# The accuracy results are then graphed, with values of k on the x-axis and accuracies on the y-axis.
def make_graphs():
    k_vals = [num for num in range(1, 52, 2)]
    training_run_accuracies = [] # Stores the accuracy values from compute_accuracy run on the training_set
    test_run_accuracies = [] # Stores the accuracy values from compute_accuracy run on the training_set
    training_averages = [] # Stores the average training set accuracy for each k value to be graphed
    training_stdevs = [] # Stores the average test set accuracy for each k value to be graphed
    test_averages = [] # Stores the standard deviation of training set accuracy for each k value to be graphed
    test_stdevs = [] # Stores the standard deviation of test set accuracy for each k value to be graphed
    for i in range(20):
        training_data, test_data = train_test_split()
        training_run_accuracies.append([x[1] for x in compute_accuracy(training_data, training_data)])
        test_run_accuracies.append([x[1] for x in compute_accuracy(training_data, test_data)])
    
    training_accuracies_by_k_val = np.array(training_run_accuracies).T.tolist()
    test_accuracies_by_k_val = np.array(test_run_accuracies).T.tolist()
    
    for k in range(len(k_vals)):
        training_averages.append(np.mean(training_accuracies_by_k_val[k]))
        training_stdevs.append(np.std(training_accuracies_by_k_val[k]))
        test_averages.append(np.mean(test_accuracies_by_k_val[k]))
        test_stdevs.append(np.std(test_accuracies_by_k_val[k]))

    # Creates a plot showcasing the average accuracies for each k value among the 20 trials when using the training data as the testing data
    plt.errorbar(k_vals, training_averages, yerr=training_stdevs)
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy Over Training Data')
    plt.title('Training Set Accuracies per k-Value')
    plt.savefig('Plots/training_set_accuracies_plot')

    plt.clf()
    
    # Creates a plot showcasing the average accuracies for each k value among the 20 trials when using the test data as the testing data
    plt.errorbar(k_vals, test_averages, yerr=test_stdevs)
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy Over Test Data')
    plt.title('Test Accuracies per k-Value')
    plt.savefig('Plots/test_set_accuracies_plot')

    plt.clf()

# Creates the graphs and saves them
make_graphs()
    
