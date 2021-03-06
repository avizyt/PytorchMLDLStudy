from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile


class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on different
        univariate feature selection models from sklearn.

        :param n_features: SelectPercentile if float else SelectBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        if problem_type == "classification":
            valid_scoring = {
                'f_classif': f_classif,
                'chi2': chi2,
                'mutul_info_classif': mutual_info_classif
            }
        else:
            valid_scoring = {
                'f_regression': f_regression,
                "mutual_info_regression": mutual_info_regression
            }

        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise ValueError("Invalid scoring method")

        # if n_features in int, we use selectkbest
        # if n_features in float, we use selectpercentile
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")

    # same fit function
    def fit(self, X, y):
        return self.selection.fit(X,y)

    # same transform function
    def transform(self, X):
        return self.selection.transform(X)

    # same fit transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)
