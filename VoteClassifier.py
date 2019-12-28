import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def importdata():
    data = pd.read_csv('house-votes-84.data.csv', sep=',')
    data = data.replace(['y', 'n'], [1, 0])
    majority = data.mode()
    print(majority)
    for i in range(17):
        data.iloc[:, i] = data.iloc[:, i].replace('?', majority.iloc[0, i])

    return data


def splitdataset(data, train_size):
    X = data.values[:, 1:]
    Y = data.values[:, 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size)
    return X_train, X_test, Y_train, Y_test


def learn(X_train, Y_train):
    dt = DecisionTreeClassifier(criterion="entropy")
    dt.fit(X_train, Y_train)
    return dt

def graph(X, Y, X_name, Y_name):
    plt.plot(X, Y)
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.show()


def main():
    data = importdata()
    X = data.values[:, 1:]
    Y = data.values[:, 0]

    print('Training dataset size : 25 % \n')
    for i in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75)
        dt = learn(X_train, Y_train)
        Y_predicted = dt.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predicted)
        print('Accuracy =', accuracy * 100, '%')
        print('Total number of nodes:', dt.tree_.node_count, '\n')
    print('------------------------------------------------------\n')

    accuracy_means = []
    trainsets_size = []
    trees_size_means = []
    test_size = 0.7
    for i in range(5):
        accuracy_list = []
        tree_sizes = []
        print('Training dataset size :', (1 - test_size) * 100, '%\n')
        for j in range(5):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
            dt = learn(X_train, Y_train)
            Y_predicted = dt.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_predicted)
            print('Accuracy =', accuracy * 100, '%')
            print('Total number of nodes:', dt.tree_.node_count, '\n')
            accuracy_list.append(accuracy)
            tree_sizes.append(dt.tree_.node_count)

        accuracy_mean = np.mean(accuracy_list)
        accuracy_max = max(accuracy_list)
        accuracy_min = min(accuracy_list)
        print('Accuracy Mean =', accuracy_mean * 100, '%')
        print('Accuracy Max =', accuracy_max * 100, '%')
        print('Accuracy Min =', accuracy_min * 100, '%')

        trees_size_mean = np.mean(tree_sizes)
        trees_size_max = max(tree_sizes)
        trees_size_min = min(tree_sizes)
        print('Tree size Mean =', trees_size_mean)
        print('Tree size Max =', trees_size_max)
        print('Tree size Min =', trees_size_min)
        print('---------------------------------------------------------\n')

        accuracy_means.append(accuracy_mean * 100)
        trainsets_size.append((1 - test_size) * 100)
        trees_size_means.append(trees_size_mean)
        test_size -= 0.1

    graph(accuracy_means, trainsets_size, 'Accuracy', 'Training set size')
    graph(trees_size_means, trainsets_size, 'Tree size', 'Training set size')


if __name__ == "__main__":
    main()