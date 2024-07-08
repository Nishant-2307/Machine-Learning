"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""
import numpy as np
import pandas as pd
from scipy.special import xlogy

def check_ifreal(X: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return X.dtype.name != 'category'


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    vals = Y.value_counts(normalize=True)
    return -np.sum(xlogy(vals, vals))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    vals = Y.value_counts(normalize=True)
    return 1 - np.sum(np.square(vals))


def information_gain(Y: pd.Series, attr: pd.Series, criterion) -> float:
    """
    Function to calculate the information gain
    """

    criterion_func_map = {
        'information_gain': entropy,
        'gini_index': gini_index,
        'mse': np.var
    }

    func = criterion_func_map[criterion]
    value_before = func(Y)
    split_value = None
    if check_ifreal(attr):
        split_value = opt_split_value(pd.DataFrame(attr), Y, attr.name)
        value_after = Y.groupby(attr <= split_value,observed=False).apply(lambda group: len(group) / len(Y) * func(group)).sum()
    else:
        value_after = Y.groupby(attr,observed=False).apply(lambda group: len(group) / len(Y) * func(group)).sum()

    return (value_before - value_after, split_value)


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    y = y if check_ifreal(y) else y.cat.codes
    scores = {feature: information_gain(y, X[feature], criterion) for feature in features}

    key = max(scores, key=lambda value: scores[value][0])
    return key, scores[key][0], scores[key][1]


def real_variance(X: pd.DataFrame, y: pd.Series, value: np.float64 , attribute):
    """
    Function to calculate the weighted variance
    """

    mask = (X[attribute] <= value)
    var_left = np.var(y[mask]) * len(y[mask])
    var_right = np.var(y[~mask]) * len(y[~mask])
    return var_left + var_right


def opt_split_value(X: pd.DataFrame, y: pd.Series, attribute):
    """
    Function to find the optimal split value for a given attribute.

    X: Input features (DataFrame)
    y: Output values (Series)
    attribute: Attribute to split upon
    criterion: Splitting criterion ('information_gain' for discrete output, 'mse' for real output)

    return: Optimal split value
    """

    X = X.sort_values(by=[attribute])
    check_values = [(X[attribute].iloc[i] + X[attribute].iloc[i+1]) / 2 for i in range(X.shape[0]-1)]

    y = y if check_ifreal(y) else y.cat.codes
    min_var = float('inf')
    optimal_value = None

    for value in check_values:
        var = real_variance(X, y, value, attribute)
        if var < min_var:
            min_var = var
            optimal_value = value

    return optimal_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value=None):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if not check_ifreal(X[attribute]):
        unique_values = np.array(X[attribute].unique())
        return [(X[X[attribute] == val], y[X[attribute] == val]) for val in unique_values], unique_values
    else:
        mask = (X[attribute] <= value)
        return [(X[mask], y[mask]), (X[~mask], y[~mask])], value
