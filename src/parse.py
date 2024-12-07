import pandas as pd
from sklearn.model_selection import train_test_split

def parse_data(filename):
    """
    Parse the data from the csv file
    """
    try:
        open(filename, "r")
    except FileNotFoundError:
        print(f"[E] File {filename} not found")
        exit()

    data = pd.read_csv(filename, sep=";")
    header = [x.strip() for x in data.columns.to_list()]
    data = data.to_numpy()
    return data, header

# print(parse_data("../Dataset/logfile.csv", 2)[:10])

def split_data(data, njoint, dimensions=2, test_size=0.2, random_state=42, consider_orientation=False, consider_sincos=True, header=None, train_test=True):
    """
    Split the data into train and test
    """

    # pos, cos, sin
    features = 3 if consider_sincos else 1
    attributes = features*njoint

    X = data[:, :attributes]
    if header is not None:
        print(f"[V] Splitting {header[:attributes]} as inputs")

    if not consider_sincos:
        attributes += 2*njoint

    if consider_orientation:
        # Consider quaternion orientation as well
        y = data[:, attributes:]
        if header is not None:
            print(f"[V] Splitting {header[attributes:]} as outputs\n")
    else:
        # Consider x, y, z coordinates only
        y = data[:, attributes:attributes+dimensions]
        if header is not None:
            print(f"[V] Splitting {header[attributes:attributes+dimensions]} as outputs\n")
    
    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    else:
        return X, y