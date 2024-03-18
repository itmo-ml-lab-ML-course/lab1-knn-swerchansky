from enum import Enum
from math import sqrt, pi, exp
from sklearn.neighbors import NearestNeighbors


class Kernel(Enum):
    UNIFORM = lambda x: 0.5 if -1 < x < 1 else 0
    TRIANGULAR = lambda x: max(0, 1 - abs(x))
    EPANECHNIKOV = lambda x: max(0, 0.75 * (1 - x ** 2))
    GAUSSIAN = lambda x: 1 / sqrt(2 * pi) * exp(- (x ** 2 / 2))

    def __init__(self, func):
        self.apply = func


class WindowType(Enum):
    FIXED = 1
    VARIABLE = 2


class Metric(Enum):
    MANHATTAN = 1
    EUCLIDEAN = 2
    COSINE = 3


class KNN:
    def __init__(
            self,
            window_param: int,
            window_type: WindowType,
            kernel: Kernel,
            metric: Metric,
    ):
        self.window_type = window_type
        self.kernel = kernel
        self.metric = metric
        if window_type == WindowType.FIXED:
            self.h = window_param
        else:
            self.k = window_param

    def fit(self, X, y, w):
        self.x_train = X
        self.y_train = y
        self.w = w
        self.count_class = len(self.y_train.value_counts())
        return self

    def predict(self, X):
        predictions = []
        all_distances, all_classes, all_weights = self.search_neighbors(X)
        for i in range(len(X)):
            distances, classes, weights = all_distances[i], all_classes[i], all_weights[i]
            scores = [0 for _ in range(self.count_class)]
            for i in range(len(distances) - 1):
                kernel_arg = distances[i] / (self.h if self.window_type == WindowType.FIXED else distances[-1])
                scores[classes[i]] += self.kernel(kernel_arg) * weights[i]
            predictions.append(scores.index(max(scores)))
        return predictions

    def search_neighbors(self, X):
        neighbors_count = self.k + 1 if self.window_type == WindowType.VARIABLE else min(int(sqrt(len(self.x_train))),
                                                                                         len(self.x_train) - 1)
        neighbours = NearestNeighbors(metric=self.metric.name.lower(), n_neighbors=neighbors_count, n_jobs=-1)
        neighbours.fit(self.x_train)
        distances, id = neighbours.kneighbors(X, n_neighbors=neighbors_count)
        classes, weights = [], []
        for i in id:
            classes.append(self.y_train.iloc[i].to_list())
            weights.append([self.w[j] for j in i])
        return (distances, classes, weights)
