import pickle
from functools import partial
from typing import Callable, Dict

import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class InvalidTargetException(Exception):
    """Exception para codificação inválida"""

    pass


def linear_kernel(x: np.array, y: np.array):
    x_dash = y * x
    return x_dash @ x_dash.T


def polinomial_kernel(x, y, d):
    m, _ = x.shape

    k = np.zeros((m, m))
    h = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            k[i, j] = ((x[i, :] @ x[j, :].T) + 1) ** d
            h[i, j] = k[i, j] * y[i] * y[j]

    return h


# noinspection PyMethodOverriding
class SVM(BaseEstimator):
    def __init__(
        self, C: float = 10, kernel: Callable = linear_kernel, degree: int = None
    ):
        self.C = C
        self.degree = degree
        self.kernel = kernel
        self.kernel_name = kernel.__name__

        # parametros do modelo
        self.support_vectors = None
        self.support_vectors_y = None
        self.bias = None
        self.w = None

    @staticmethod
    def validate_target(y: np.array):
        """Variável alvo deve ser binária, codificada como 1 e -1"""
        unique_values = set(np.unique(y.astype(int)))
        return all(u in {1, -1} for u in unique_values)

    def solve_qp(self, k: np.array, x: np.array, y: np.array):
        m, _ = x.shape

        p = cvxopt_matrix(k)
        q = cvxopt_matrix(-np.ones((m, 1)))
        g = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        a = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        solution = cvxopt_solvers.qp(p, q, g, h, a, b)
        lagrange_multipliers = np.array(solution["x"])

        support_idx = (lagrange_multipliers > 1e-4).flatten()

        self.support_vectors = x[support_idx]
        self.support_vectors_y = y[support_idx]

        w = ((y * lagrange_multipliers).T @ x).reshape(-1, 1)
        b = self.support_vectors_y - np.dot(self.support_vectors, w)

        return w, b

    def fit(self, x, y):
        logger.info(
            f"Fitting SVC with params: C = {self.C}, kernel = {self.kernel_name}, d = {self.degree}"
        )

        if not self.validate_target(y):
            raise InvalidTargetException("Targets devem estar codificados como -1 e 1")

        if self.kernel_name == "polinomial_kernel":
            self.kernel = partial(self.kernel, d=self.degree)

        y = y.reshape(-1, 1).astype(np.float64)
        k = self.kernel(x, y)
        w, b = self.solve_qp(k, x, y)

        self.w = w
        self.bias = b[0] if len(b) else [0]

        return self

    def predict(self, x):
        if self.kernel_name == "linear_kernel":
            return np.sign(x @ self.w + self.bias)
        ...

    def score(self, x: np.array, y: np.array) -> float:
        y_pred = self.predict(x)
        return np.sum(y == y_pred) / len(y)

    def __repr__(self):
        return f"SVC(kernel={self.kernel_name}, C={self.C}, d={self.degree})"


def load_dataset(path: str):
    df = pd.read_parquet(path).dropna()
    x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    y = 2 * y - 1
    return x, y.astype(np.int64)


def load_lbp():
    return load_dataset("data/preprocessed/feature_matrix_lbp.parquet")


def load_hog():
    return load_dataset("data/preprocessed/feature_matrix_hog.parquet")


def make_cross_val(x, y):
    params = {
        "clf__C": [1, 10, 100, 1000],
        "clf__degree": [2, 3, 4],
        "clf__kernel": [linear_kernel],
    }

    est = Pipeline([("scl", StandardScaler()), ("clf", SVM())])

    cv = GridSearchCV(
        estimator=est, param_grid=params, n_jobs=-1, cv=5, error_score="raise"
    )
    cv.fit(x, y)

    return cv.best_estimator_, cv.cv_results_


def save_model_results(
    dataset_name: str,
    classification_report_text: str,
    results: Dict,
    model: BaseEstimator,
):
    with open(f"execucao/{dataset_name}-classification_report.txt", "w") as f:
        f.write(classification_report_text)

    with open(f"execucao/{dataset_name}-results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"execucao/{dataset_name}-model.pkl", "wb") as f:
        pickle.dump(model, f)


def main():
    datasets = [("LBP", load_lbp), ("HOG", load_lbp)]

    for dataset_name, loader in datasets:
        x, y = loader()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        best_model, results = make_cross_val(x_train, y_train)

        y_pred = best_model.predict(x_test)

        class_report = classification_report(y_test, y_pred)

        save_model_results(dataset_name, class_report, results, best_model)


if __name__ == "__main__":
    main()
