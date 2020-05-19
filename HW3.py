#!/usr/bin/env python
# coding: utf-8

# ## HW3: Decision Tree and Random Forest
# In hw3, you need to implement decision tree and random forest by using only numpy, then train your
# implemented model by the provided dataset and test the performance with testing data
#
# Please note that only **NUMPY** can be used to implement your model, you will get no points by
# simply calling sklearn.tree.DecisionTreeClassifier
# 
# by Alexander Kjeldsen

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from math import log2
import random
from matplotlib import pyplot as plt


data = load_breast_cancer()
feature_names = data['feature_names']

x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

# ## Question 1
# Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute
# the Entropy and Gini Index of provided data. Please use the formula from page 666 on the textbook


def gini(sequence):
    sequence = np.array(sequence)
    rt = np.unique(sequence)
    if len(rt) <= 1:
        return 0
    pr = []
    for v in rt:
        pr.append(sequence.tolist().count(v) / len(sequence))
    return 1 - np.sum(list(map(lambda x: x*x , pr)))

def entropy(sequence):
    sequence = np.array(sequence)
    rt = np.unique(sequence)
    if len(rt) <= 1:
        return 0
    pr = []
    for v in rt:
        pr.append(sequence.tolist().count(v) / len(sequence))
    return -np.sum(list(map(lambda x: x*log2(x), pr)))


# 1 = class 1,
# 2 = class 2
data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])


print("Gini of data is ", gini(data))  # 0.16969
print("Entropy of data is ", entropy(data))  # 0.65548


# ## Question 2
# Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained
# the model by the given arguments, and print the accuracy score on the test data. You should implement
# two arguments for the Decision Tree algorithm
# 1. **Criterion**: The function to measure the quality of a split. Your model should support “gini” for
# the Gini impurity and “entropy” for the information gain.
# 2. **Max_depth**: The maximum depth of the tree. If Max_depth=None, then nodes are expanded until all
# leaves are pure. Max_depth=1 equals to split data once
#


class DecisionTree():

    def __init__(self, criterion='gini', max_depth=None):

        self.criterion = criterion
        self.max_depth = max_depth
        self.label = None
        self.impurity = None
        self.left = None
        self.right = None
        self.best_feature_index = None
        self.impurity = None

    def fit(self, xs, ys):
        xs = np.array(x_train)
        ys = np.array(y_train).take(indices=0, axis=1)
        self.model(xs, ys, self.max_depth)

    def model(self, xs, ys, depth):
        self.best_information_gain = 0
        self.best_threshold = -1
        self.best_feature_index = -1
        self.impurity = gini(ys) if self.criterion == 'gini' else entropy(ys)
        self.n = np.size(ys)

        for feature_index in range(len(xs.take(indices=0, axis=0))):
            thresholds = (np.unique(xs.take(indices=feature_index, axis=1)))
            for threshold in thresholds:
                candidate_child_1 = ys[xs[:, feature_index] <= threshold]
                impurity_child_1 = gini(
                    candidate_child_1) if self.criterion == 'gini' else entropy(candidate_child_1)

                candidate_child_2 = ys[xs[:, feature_index] > threshold]
                impurity_child_2 = gini(
                    candidate_child_2) if self.criterion == 'gini' else entropy(candidate_child_2)

                self.impurity_of_split = impurity_child_1 + impurity_child_2
                information_gain = self.impurity - self.impurity_of_split
                if information_gain > self.best_information_gain:
                    self.best_information_gain = information_gain
                    self.best_purity = self.impurity
                    self.best_feature_index = feature_index
                    self.best_threshold = threshold
                    self.best_impurity_child_1 = impurity_child_1
                    self.best_impurity_child_2 = impurity_child_2

        #print (self.best_feature_index, self.best_information_gain, threshold)
        #print("depth", self.max_depth)

        if self.max_depth <= 0 or self.best_information_gain<=0:
            self.best_feature_index=None
            return

        self.left = DecisionTree(criterion=self.criterion, max_depth=depth-1)
        self.child_1 = xs[xs[:, self.best_feature_index] <= self.best_threshold]
        self.child_1_ys = ys[xs[:, self.best_feature_index] <= self.best_threshold]
        self.left.label = np.bincount(self.child_1_ys).argmax()

        self.right = DecisionTree(criterion=self.criterion, max_depth=depth-1)
        self.child_2 = xs[xs[:, self.best_feature_index] > self.best_threshold]
        self.child_2_ys = ys[xs[:, self.best_feature_index] > self.best_threshold]
        self.right.label = np.bincount(self.child_2_ys).argmax()

        self.left.model(self.child_1, self.child_1_ys, depth-1)
        self.right.model(self.child_2, self.child_2_ys, depth-1)


def predict(object, data):
    data = np.array(data)
    if object.best_feature_index is not None:
        if data[object.best_feature_index] <= object.best_threshold:
            return predict(object.left, data)
        else:
            return predict(object.right, data)
    else:
        return object.label


def test(model, xs, ys, text):
    xs = np.array(xs)
    ys = np.array(ys)
    correct = 0
    for i in range(len(ys)):
        if(predict(model, np.array(xs).take(indices=i, axis=0)) == ys[i]):
            correct = correct + 1
    print(text, "test:", correct/float(len(y_test)))


# ### Question 2.1
# Using Criterion=‘gini’, showing the accuracy score of test data by Max_depth=3 and Max_depth=10, respectively.
#
clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth3.fit(x_train, y_train)
test(clf_depth3, x_test, y_test, "depth 3")

clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
clf_depth10.fit(x_train, y_train)
test(clf_depth10, x_test, y_test, "depth 10")

# ### Question 2.2
# Using Max_depth=3, showing the accuracy score of test data by Criterion=‘gini’ and Criterion=’entropy’, respectively.
#


clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_gini.fit(x_train, y_train)
test(clf_gini, x_test, y_test, "gini")


clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
clf_entropy.fit(x_train, y_train)
test(clf_entropy, x_test, y_test, "entropy")

# - Note: All of your accuracy scores should over 0.9
# - Note: You should get the same results when re-building the model with the same arguments,  no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
#

# ## Question 3
# Plot the [feature importance](https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/ )
# of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
#
# - You can simply plot the feature counts for building tree without normalize the importance
#
# ![image](https://i2.wp.com/sefiks.com/wp-content/uploads/2020/04/c45-fi-results.jpg?w=481&ssl=1 )

def calc_importance(object, n, level=1):
    if level == 1:
        return object.n/n*(object.impurity
                           -object.right.n/float(object.n)*object.right.impurity
                           -object.left.n/float(object.n)*object.left.impurity )
    else :
        return calc_importance(object.left, level-1) - calc_importance(object.right, level-1)

def feature_importance(object, n, arr):
    if object.best_feature_index is not None:
        #print(object.best_feature_index, calc_importance(object, n))
        arr[object.best_feature_index] = calc_importance(object, n)
        feature_importance(object.left, n, arr)
        feature_importance(object.right, n, arr)
        return arr


arr = feature_importance(clf_depth10, clf_depth10.n, np.zeros(shape=30))

plt.style.use('ggplot')
x = feature_names[arr[:]>0]
energy = arr[arr[:]>0]
x_pos = [i for i, _ in enumerate(x)]
plt.barh(x_pos, energy)
plt.xlabel("feature names")
plt.ylabel("importance index")
plt.title("Feature Importance")
plt.yticks(x_pos, x, fontsize=5)
plt.show()

# ## Question 4
# implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement two arguments for the Random Forest.
#
# 1. **N_estimators**: The number of trees in the forest.
# 2. **Max_features**: The number of random select features to consider when looking for the best split
# 3. **Bootstrap**: Whether bootstrap samples are used when building tree
#

dat = np.array(x_train)

r = np.round(random.random()*len(dat.take(indices=0, axis=0)) + 1 )

random_forrest = random.sample(list(dat), int(r))

random.sample(list(x_test), 10)

xs = np.array(x_train)
xs
random.sample(list(xs), 10)



#%%

"""
class RandomTree():
    def __init__(self, xs, ys, max_features):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.max_features = max_features
        self.criterion = 'gini'
        self.label = None

    def fit(self):

        self.best_information_gain = 0

        # self.xs_subset = self.xs.take(indices=self.random_class_indices, axis=1)
        self.impurity = gini(self.ys) if self.criterion == 'gini' else entropy(self.ys)


        for col in self.random_class_indices:
            self.thresholds = np.unique(self.xs.take(indices=self.random_class_indices, axis=0))
            for threshold in self.thresholds:
                candidate_child_1 = self.ys[self.xs[:, col] <= threshold]
                impurity_child_1 = gini(
                    candidate_child_1) if self.criterion == 'gini' else entropy(candidate_child_1)

                candidate_child_2 = self.ys[self.xs[:, col] > threshold]
                impurity_child_2 = gini(
                    candidate_child_2) if self.criterion == 'gini' else entropy(candidate_child_2)

                self.impurity_of_split = impurity_child_1 + impurity_child_2
                information_gain = self.impurity - self.impurity_of_split
                if information_gain > self.best_information_gain:
                    self.best_information_gain = information_gain
                    self.best_purity = self.impurity
                    self.best_feature_index = col
                    self.best_threshold = threshold
                    self.best_impurity_child_1 = impurity_child_1
                    self.best_impurity_child_2 = impurity_child_2

        #print (self.best_feature_index, self.best_information_gain, threshold)
        #print("depth", self.max_depth)

        if self.best_information_gain<=0:
            self.best_feature_index=None
            return

        self.child_1 = self.xs[self.xs[:, self.best_feature_index] <= self.best_threshold]
        self.child_1_ys = self.ys[self.xs[:, self.best_feature_index] <= self.best_threshold]
        self.left = RandomTree(self.child_1, self.child_1_ys, self.max_features)
        self.left.label = np.bincount(self.child_1_ys).argmax()


        self.child_2 = self.xs[self.xs[:, self.best_feature_index] > self.best_threshold]
        self.child_2_ys = self.ys[self.xs[:, self.best_feature_index] > self.best_threshold]
        self.right = RandomTree(self.child_2, self.child_2_ys, self.max_features)
        self.right.label = np.bincount(self.child_2_ys).argmax()

        self.left.fit()# self.child_1, self.child_1_ys, depth-1)
        self.right.fit()# self.child_2, self.child_2_ys, depth-1)
"""
#%%

class RandomDecisionTree():

    def __init__(self, criterion='gini', max_depth=None, max_features=5):
        self.max_features = max_features
        self.criterion = criterion
        self.max_depth = max_depth
        self.label = None
        self.impurity = None
        self.left = None
        self.right = None
        self.best_feature_index = None
        self.impurity = None

    def fit(self, xs, ys):
        xs = np.array(xs)
        ys = np.array(ys).take(indices=0, axis=1)
        self.model(xs, ys, self.max_depth)

    def model(self, xs, ys, depth):
        self.best_information_gain = 0
        self.best_threshold = -1
        self.best_feature_index = -1
        self.impurity = gini(ys) if self.criterion == 'gini' else entropy(ys)
        self.n = np.size(ys)
        indices = (np.random.rand(int(self.max_features))*len(xs.take(indices=1, axis=0))).astype(int)
        self.random_class_indices = indices

        for feature_index in self.random_class_indices:
            thresholds = (np.unique(xs.take(indices=feature_index, axis=1)))
            for threshold in thresholds:
                candidate_child_1 = ys[xs[:, feature_index] <= threshold]
                impurity_child_1 = gini(
                    candidate_child_1) if self.criterion == 'gini' else entropy(candidate_child_1)

                candidate_child_2 = ys[xs[:, feature_index] > threshold]
                impurity_child_2 = gini(
                    candidate_child_2) if self.criterion == 'gini' else entropy(candidate_child_2)

                self.impurity_of_split = impurity_child_1 + impurity_child_2
                information_gain = self.impurity - self.impurity_of_split
                if information_gain > self.best_information_gain:
                    self.best_information_gain = information_gain
                    self.best_purity = self.impurity
                    self.best_feature_index = feature_index
                    self.best_threshold = threshold
                    self.best_impurity_child_1 = impurity_child_1
                    self.best_impurity_child_2 = impurity_child_2

        #print (self.best_feature_index, self.best_information_gain, threshold)

        if self.max_depth is not None:
            self.max_depth -= 1
            if self.max_depth <= 0 :
                return

        if self.best_information_gain<=0 or self.best_feature_index == -1 :
            self.best_feature_index=None
            return


        self.left = RandomDecisionTree(criterion=self.criterion, max_depth=self.max_depth, max_features=self.max_features)
        self.child_1 = xs[xs[:, self.best_feature_index] <= self.best_threshold]
        self.child_1_ys = ys[xs[:, self.best_feature_index] <= self.best_threshold]
        self.left.label = np.bincount(self.child_1_ys).argmax()

        self.right = RandomDecisionTree(criterion=self.criterion, max_depth=self.max_depth, max_features=self.max_features)
        self.child_2 = xs[xs[:, self.best_feature_index] > self.best_threshold]
        self.child_2_ys = ys[xs[:, self.best_feature_index] > self.best_threshold]
        self.right.label = np.bincount(self.child_2_ys).argmax()

        # len(xs.take(indices=0, axis=0))
        if len(self.child_1_ys) > 1:
            self.left.model(self.child_1, self.child_1_ys, depth)
        else :
            self.left.best_feature_index = None

        if len(self.child_2_ys) > 1:
            self.right.model(self.child_2, self.child_2_ys, depth)
        else:
            self.right.best_feature_index=None


class RandomForest():
    def __init__(self, n_estimators, max_features, bootstrap=True, criterion='gini', max_depth=None, kind=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.kind = kind


    def fit(self, xs, ys):
        self.xs = np.array(xs)
        self.ys = np.array(ys).take(indices=0, axis=1)

        self.trees = []

        if self.bootstrap: # with bootstrapped dataset
            number = 1
            for i in range(self.n_estimators):
                if self.kind is not None:
                    if self.kind == 'all_features':
                        # print('all features')
                        self.trees.append(RandomDecisionTree(max_features=len(self.xs.take(indices=0, axis=0))))
                    elif self.kind == 'random_features':
                        r = int(np.random.rand()*len(self.xs.take(indices=0, axis=0)))
                        # print('random features', r)
                        self.trees.append(RandomDecisionTree(max_features=r))
                    else :
                        print ("error in kind parameter \t choose all_features or random_features")
                else:
                    # print("normal random decision tree")
                    self.trees.append(RandomDecisionTree())
            for tree in self.trees:
                s = (np.random.rand(len(xs))*len(xs)).astype(int)
                self.bootstrap_indices = s
                self.xs_boot = xs.take(indices=s, axis=0)
                self.ys_boot = ys.take(indices=s, axis=0)
                tree.fit(self.xs_boot, self.ys_boot)
                # print("\ttree finished",number, "/", len(self.trees))
                number = number + 1
        else: # without bootstrapping dataset
            for i in range(self.n_estimators):
                self.trees.append(RandomDecisionTree())
            for tree in self.trees:
                self.tree.fit(self.xs, self.ys)


def predictRandomTree(tree : RandomDecisionTree, data):
    data = np.array(data)
    if tree.best_feature_index is not None:
        if data[tree.best_feature_index] <= tree.best_threshold:
            return predict(tree.left, data)
        else:
            return predict(tree.right, data)
    else:
        return tree.label



def testRandomForrest(model : RandomForest, xs, ys, text):
    xs = np.array(xs)
    ys = np.array(ys)
    correct = 0
    for i in range(len(ys)):
        forrest_predictions = []
        for tree in model.trees:
            p = predict(tree, np.array(xs).take(indices=i, axis=0))
            if p is not None:
                forrest_predictions.append(p)

        prediction = np.bincount(forrest_predictions).argmax()
        if(prediction == ys[i]):
            correct = correct + 1
    print(text, "test:", correct/float(len(y_test)))


# ### Question 4.1
# Using Criterion=‘gini’, Max_depth=None, Max_features=sqrt(n_features), showing the accuracy score of test data by n_estimators=10 and n_estimators=100, respectively.
#


clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_10tree.fit(x_train, y_train)
testRandomForrest(clf_10tree, x_test, y_test, "clf_10tree")

clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
clf_100tree.fit(x_train, y_train)
testRandomForrest(clf_100tree, x_test, y_test, "clf_100tree")


# ### Question 4.2
# Using Criterion=‘gini’, Max_depth=None, N_estimators=10, showing the accuracy score of test data by Max_features=sqrt(n_features) and Max_features=n_features, respectively.
#


clf_all_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]), kind='all_features')
clf_all_features.fit(x_train, y_train)
testRandomForrest(clf_all_features, x_test, y_test, "clf_all_features")


clf_random_features = RandomForest(n_estimators=10, max_features=x_train.shape[1], kind='random_features')
clf_random_features.fit(x_train, y_train)
testRandomForrest(clf_random_features, x_test, y_test, "clf_random_features")


# - Note: Use majority votes to get the final prediction, you may get slightly different results when re-building the random forest model


# ## Supplementary
# If you have trouble to implement this homework, TA strongly recommend watching [this video](https://www.youtube.com/watch?v=LDRbO9a6XPU ),
# which explains Decision Tree model clearly. But don't copy code from any resources, try to finish this homework by yourself!
