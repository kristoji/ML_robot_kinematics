from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import parse

if __name__ == "__main__":

    DIM = 2
    NJOINT = 3
    ORIENTATION = True
    VALIDATION = False

    # ------------------------------------------------------------------------
    # Load the data

    noutput = DIM + (4 if ORIENTATION and DIM == 3 else 2) if ORIENTATION else DIM

    data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}.csv")
    X_train, X_test, y_train, y_test = parse.split_data(data, NJOINT, DIM, consider_orientation=ORIENTATION, header=header)

    assert noutput == y_train.shape[1]
    
    # -----------------------------------------------------------------------
    # Hyperparameter to tune
    param_grid = {
        'estimator__C': [0.01, 0.1, 1, 10, 100],
        'estimator__kernel': ['rbf', 'sigmoid'],
        'estimator__gamma': ['scale', 'auto'],
        "estimator__epsilon": [0.1, 0.2, 0.5]
    }

    # -----------------------------------------------------------------------
    # Hyperparameter tuning
    
    svr = MultiOutputRegressor(SVR())

    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best CV Accuracy: {best_score}")

    y_pred = best_estimator.predict(X_test)
    test_accuracy = best_estimator.score(X_test, y_test)

    print("\nTest Set Evaluation:")
    print(f"Test Accuracy: {test_accuracy}")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------------------------
    # Save the best model
    best_estimator.save("best_model.keras")