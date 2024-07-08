"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    # Node class for the decision tree
    def __init__(self, attribute=None, depth=None, impurity=None, split_values=None, value=None):
        self.attribute = attribute
        self.value = value
        self.split_values = split_values
        self.impurity = impurity
        self.depth = depth
        self.subnodes = {}
        self.is_leaf = False if value is None else True


@dataclass
class DecisionTree:
    # DecisionTree class for the decision tree
    # criterion won't be used for regression
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.node = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        # Base cases for recursion
        attr, impurity, split_value = opt_split_attribute(X, y, self.criterion, X.columns.tolist())
        if depth >= self.max_depth or X.shape[0] == 0 or y.nunique() == 1:
            self.node = Node(depth=depth, impurity=impurity, value=y.median() if check_ifreal(y) else y.mode()[0])
            return

        # Split the data based on the optimal attribute
        splitted_data, values = split_data(X, y, attr, split_value)

        # Create a new node and add subnodes to it
        self.node = Node(attr, depth, impurity, values)
        if check_ifreal(X[attr]):
            for value, data in zip([True, False], splitted_data):
                subtree = DecisionTree(self.criterion, self.max_depth)
                subtree.fit(data[0], data[1], depth + 1)
                self.node.subnodes[value] = subtree
        else:
            for value, data in zip(values, splitted_data):
                subtree = DecisionTree(self.criterion, self.max_depth)
                subtree.fit(data[0], data[1], depth + 1)
                self.node.subnodes[value] = subtree

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def traverse_tree(x, node:Node):
            if node.is_leaf:
                return node.value

            attr = node.attribute
            split_values = node.split_values
            value = x[attr] <= split_values if check_ifreal(x) else x[attr]
            if value in node.subnodes:
                return traverse_tree(x, node.subnodes[value].node)
            else:
                return node.value

        # Apply the helper function to each instance in the test inputs
        predictions = X.apply(traverse_tree, axis=1, args=(self.node,))
        return predictions

    def plot(self, depth=0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        node = self.node

        if node.is_leaf:
            # If the node is a leaf node
            print(f"Class {node.value}")
        else:
            # If the node is an internal node
            attribute = node.attribute
            split_values = node.split_values
            if isinstance(split_values, np.ndarray):
                # Discrete input
                for value, subnode in node.subnodes.items():
                    endline = '' if subnode.node.is_leaf else '\n'
                    print(f"{' ' * depth * 4}?(column {attribute} == {value}): ", end=endline)
                    subnode.plot(depth + 1)
            else:
                # Real input
                print(f"?({attribute} <= {split_values})")
                for value, subnode in node.subnodes.items():
                    print(f"{' ' * depth * 4}{value}: ", end="")
                    subnode.plot(depth + 1)