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


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, d=3):
    return (1 + np.dot(x, y)) ** d


# noinspection PyMethodOverriding
class SVM(BaseEstimator):
    def __init__(
        self, C: int = 10, kernel: Callable = linear_kernel, degree: int = None
    ):
        self.C = C
        self.degree = degree
        self.kernel = kernel
        self.kernel_name = kernel.__name__

        # parâmetros do modelo
        self.support_vectors = None
        self.support_vectors_y = None
        self.bias = None
        self.w = None
        self.lagrange_multipliers = None

    @staticmethod
    def validate_target(y: np.array):
        """Variável alvo deve ser binária, codificada como 1 e -1"""
        unique_values = set(np.unique(y.astype(int)))
        if all(u in {1, -1} for u in unique_values):
            return True
        raise InvalidTargetException(
            "Variável alvo deve ser binária, codificada como 1 e -1"
        )

    def solve_qp(self, x: np.array, y: np.array):
        n_samples, _ = x.shape

        k = self.kernel(x, x.T)
        p = cvxopt_matrix(np.outer(y, y) * k)
        q = cvxopt_matrix(-np.ones(n_samples))
        g = cvxopt_matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt_matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        a = cvxopt_matrix(y, (1, n_samples), "d")
        b = cvxopt_matrix(0.0)

        solution = cvxopt_solvers.qp(p, q, g, h, a, b)
        alphas = np.ravel(solution["x"])

        support_idx = alphas > 1e-4

        lagrange_multipliers = alphas[support_idx]
        support_vectors = x[support_idx]
        support_vectors_y = y[support_idx]

        return lagrange_multipliers, support_vectors, support_vectors_y

    def estimate_bias(self):
        """Calcula o bias"""
        if self.kernel_name == "polynomial_kernel":
            bias = np.mean(
                self.support_vectors_y
                - sum(
                    self.lagrange_multipliers
                    * self.support_vectors_y
                    * self.kernel(self.support_vectors, self.support_vectors.T)
                )
            )
        elif self.kernel_name == "linear_kernel":
            bias = np.median(self.support_vectors_y - (self.support_vectors @ self.w))
        return bias

    def estimate_coefficients(self):
        """Calcula os coeficientes para o caso do kernel linear"""
        return self.lagrange_multipliers * self.support_vectors_y @ self.support_vectors

    def setup_kernel(self):
        """Realiza operações de configuração específicas para cada kernel"""
        if self.kernel_name == "polynomial_kernel":
            self.kernel = partial(self.kernel, d=self.degree)

    def fit(self, x, y):
        """Ajusta os parâmetros do modelo"""
        logger.info(f"Fitting model {self}")
        self.setup_kernel()

        SVM.validate_target(y)

        (
            self.lagrange_multipliers,
            self.support_vectors,
            self.support_vectors_y,
        ) = self.solve_qp(x, y)

        self.w = self.estimate_coefficients()
        self.bias = self.estimate_bias()

        return self

    def predict(self, x):
        """Recebe novos dados de entrada e retorna as predições {1, -1}"""

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
        """Recebe um conjunto de dados x e a variável y e retorna a acurácia do modelo neste conjunto"""
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
        "clf__C": [10, 100, 1000],
        "clf__degree": [2, 3, 4],
        "clf__kernel": [linear_kernel, polynomial_kernel],
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
    preds_df: pd.DataFrame,
):
    """Salva resultados dos experimentos"""
    with open(f"execucao/{dataset_name}-classification_report.txt", "w") as f:
        f.write(classification_report_text)

    with open(f"execucao/{dataset_name}-results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"execucao/{dataset_name}-model.pkl", "wb") as f:
        pickle.dump(model, f)

    preds_df.to_csv(f"execucao/{dataset_name}-preds.csv", index=False)


def main():
    """Executa toda pipeline de treinamento para todos os conjuntos de dados"""
    datasets = [("LBP", load_lbp), ("HOG", load_lbp)]

    for dataset_name, loader in datasets:
        x, y = loader()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        best_model, results = make_cross_val(x_train, y_train)

        y_pred = best_model.predict(x_test)

        class_report = classification_report(y_test, y_pred)
        preds_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

        save_model_results(dataset_name, class_report, results, best_model, preds_df)


if __name__ == "__main__":
    main()
