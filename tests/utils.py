import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from interpret.glassbox import ExplainableBoostingClassifier

from modelsight._typing import CVModellingOutput, CVScheme, Estimator, ArrayLike, SeedType


def get_model(seed: SeedType) -> Estimator:
    return ExplainableBoostingClassifier(random_state=seed,
                                         interactions=6,
                                         learning_rate=0.02,
                                         min_samples_leaf=5,
                                         n_jobs=4)


def get_calibrated_model(model: Estimator, 
                         X: ArrayLike, y: ArrayLike) -> Estimator:
    """Calibrate an already fitted model using data X, y.

    Parameters
    ----------
    model : Estimator
        Already fitted model that should be calibrated
    X : ArrayLike
        Design matrix of n features and m observations.
    y : ArrayLike
        Array of m ground-truths

    Returns
    -------
    Estimator
        The calibrated model
    """
    calib_model = CalibratedClassifierCV(estimator=model,
                                         method="sigmoid",
                                         n_jobs=10,
                                         cv="prefit")
    calib_model.fit(X, y)
    
    return calib_model


def get_feature_selector(seed: SeedType) -> Estimator:
    fs = RandomForestClassifier(random_state=seed,
                                n_estimators=10,
                                max_depth=3,
                                n_jobs=4,
                                min_samples_leaf=2,
                                min_samples_split=3)
    return fs


def select_features(X: ArrayLike,
                    y: ArrayLike,
                    selector: Estimator,
                    cv: CVScheme,
                    scale: bool,
                    frac: float) -> set[str]:
    """Custom feature selection using a feature selector (in our case
    a Random Forest) and cross-validation. We select all features that
    are associated with a Gini impurity reduction at least 25% greater
    than the mean value.

    Parameters
    ----------
    X : ArrayLike
        An array of features values
    y : ArrayLike
        An array of ground truths
    selector : Estimator
        The feature selector. In our case is a Random Forest Classifier
        and we will use the built-in feature importances.
    cv : CVScheme
        A cross-validation scheme for selecting features.
    scale: bool
        Whether data should be scaled according to a StandardScaler
    frac: float
        Fraction of mean importance 

    Returns
    -------
    The set union of all selected features
    """
    inner_feature_sets = dict()

    for ji, (train_idxs_inner, _) in enumerate(cv.split(X, y)):
        # print(f"Inner split no. {ji+1}")
        inner_split = f"inner_split_{ji}"
        inner_feature_sets[inner_split] = set()

        # select inner training and validation folds
        X_train_inner = X.iloc[train_idxs_inner, :].copy()
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        y_train_inner = y.iloc[train_idxs_inner]

        # X_val_inner, y_val_inner = X.iloc[val_idxs_inner].copy(), y[val_idxs_inner]

        if scale:
            numeric_cols = X_train_inner.select_dtypes(
                include=[np.float64, np.int64]).columns.tolist()
            scaler = StandardScaler()
            X_train_inner.loc[:, numeric_cols] = scaler.fit_transform(
                X_train_inner.loc[:, numeric_cols])
            # X_val_inner.loc[:, numeric_cols] = scaler.transform(X_val_inner.loc[:, numeric_cols])

        # feature selection
        selected_features = set()

        feat_selector = clone(selector)
        feat_selector.fit(X_train_inner, y_train_inner)

        selected_features = list(map(lambda t: t[0], list(filter(
            lambda w: w[1] > frac*feat_selector.feature_importances_.mean(), zip(X.columns, feat_selector.feature_importances_)))))

        if not selected_features:
            continue

        # Check that each output feature is present in the dataset's columns.
        diff = set.difference(set(selected_features),
                              set(X_train_inner.columns))
        if diff:
            raise ValueError(
                F"Could not find features {DIFF} in the dataframe.")

        if not isinstance(selected_features, set):
            selected_features = set(selected_features)

        inner_feature_sets[inner_split] = selected_features

    return sorted(set.union(*inner_feature_sets.values()))
