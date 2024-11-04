import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_data(filename):
    """
    Parse the data from the csv file
    """
    data = pd.read_csv(filename, sep=";")
    header = data.columns.to_list()
    data = data.to_numpy()
    return data, header

# print(parse_data("../Dataset/logfile.csv", 2)[:10])

def split_data(data, njoint, dimensions=2, test_size=0.2, random_state=42, consider_orientation=False, header=None):
    """
    Split the data into train and test
    """

    # pos, cos, sin
    features = 3
    attributes = features*njoint

    X = data[:, :attributes]
    if header is not None:
        print(f"[V] Splitting {header[:attributes]} as inputs")

    if consider_orientation:
        # Consider quaternion orientation as well
        y = data[:, attributes:]
        if header is not None:
            print(f"[V] Splitting {header[attributes:]} as outputs")
    else:
        # Consider x, y, z coordinates only
        y = data[:, attributes:attributes+dimensions]
        if header is not None:
            print(f"[V] Splitting {header[attributes:attributes+dimensions]} as attributes")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



# data = parse_data("../Dataset/logfile.csv")
# print(data[:10])
# print()
# X_train, X_test, y_train, y_test = split_data(data[:10], 2)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)