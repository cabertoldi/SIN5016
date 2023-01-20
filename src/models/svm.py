import pickle
from functools import partial
from typing import Callable, Dict

import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


class InvalidTargetException(Exception):
    """Exception para codificação inválida"""

    pass


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, d=3):
    return (1 + np.dot(x, y)) ** d


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
        self.lagrange_multipliers = None

    @staticmethod
    def validate_target(y: np.array):
        """Variável alvo deve ser binária, codificada como 1 e -1"""
        unique_values = set(np.unique(y.astype(int)))
        return all(u in {1, -1} for u in unique_values)

    def solve_qp(self, k: np.array, x: np.array, y: np.array):
        n_samples, _ = x.shape

        p = cvxopt_matrix(np.outer(y, y) * k)
        q = cvxopt_matrix(-np.ones(n_samples))
        g = cvxopt_matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt_matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        a = cvxopt_matrix(y, (1, n_samples), "d")
        b = cvxopt_matrix(0.0)

        solution = cvxopt_solvers.qp(p, q, g, h, a, b)
        alphas = np.ravel(solution["x"])

        support_idx = alphas > 1e-4

        ind = np.arange(len(alphas))[
            support_idx
        ]  # lista com indices dos vetores suporte

        self.lagrange_multipliers = alphas[support_idx]
        self.support_vectors = x[support_idx]
        self.support_vectors_y = y[support_idx]

        logger.info("calculating bias")

        logger.info("calculating w")
        self.w = (
            self.lagrange_multipliers * self.support_vectors_y @ self.support_vectors
        )
        self.bias = np.median(self.support_vectors_y - (self.support_vectors @ self.w))

        if self.kernel_name == "polynomial_kernel":
            self.bias = np.mean(
                self.support_vectors_y
                - sum(
                    self.lagrange_multipliers
                    * self.support_vectors_y
                    * self.kernel(self.support_vectors, self.support_vectors.T)
                )
            )

        return self.w, self.bias

    def fit(self, x, y):
        logger.info(
            f"Fitting SVC with params: C = {self.C}, kernel = {self.kernel_name}, d = {self.degree}"
        )

        if not self.validate_target(y):
            raise InvalidTargetException("Targets devem estar codificados como -1 e 1")

        if self.kernel_name == "polynomial_kernel":
            self.kernel = partial(self.kernel, d=self.degree)

        n_samples, n_features = x.shape

        # Gram matrix
        k = self.kernel(x, x.T)
        self.solve_qp(k, x, y)

        return self

    def predict(self, x):

        if self.kernel_name == "linear_kernel":
            return np.sign(x @ self.w + self.bias)

        y_pred = np.sum(
            self.lagrange_multipliers
            * self.support_vectors_y
            * self.kernel(x, self.support_vectors.T),
            axis=1,
        )

        return np.sign(y_pred + self.bias)

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


def test():
    x, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
    y = 2 * y - 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = SVM(C=10, kernel=polynomial_kernel, degree=3)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(confusion_matrix(y_test.astype(np.int64), y_pred.astype(np.int64)))
    print(classification_report(y_test.astype(np.int64), y_pred.astype(np.int64)))


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
    test()
