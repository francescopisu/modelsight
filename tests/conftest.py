
import os
import sys
import pickle
import pytest
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier

from modelsight._typing import CVModellingOutput, Estimator
from modelsight.config import settings

from utils import (
    select_features,
    get_feature_selector,
    get_calibrated_model
)

import logging
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('./src'))


class Colors:
    blue = "#4974a5"
    lightblue = "#b0c4de"
    salmon = "#ff8c69"
    lightsalmon = "#FFD1C2"
    darksalmon = "#e57e5e"

    green2 = "#82B240"
    charcoal = "#31485D"
    gray = "#A3BAC3"
    white = "#EAEBED"

    yellow = "#FFC325"
    red = "#E31B23"
    blue = "#005CAB"
    violet = "#9A348E"


class BinaryModelFactory:
    def get_model(self, model_name: str) -> Estimator:
        if model_name == "EBM":
            return ExplainableBoostingClassifier(random_state=settings.misc.seed,
                                        interactions=6,
                                        learning_rate=0.02,
                                        min_samples_leaf=5,
                                        n_jobs=-1)
        elif model_name == "LR":
            return LogisticRegression(penalty="elasticnet",
                                 solver="saga",
                                 l1_ratio=0.3,
                                 max_iter=10000,
                                 random_state=settings.misc.seed)
        elif model_name == "SVC": 
            return SVC(probability=True, 
                   class_weight="balanced", 
                   random_state=settings.misc.seed)
        elif model_name == "RF": 
            return RandomForestClassifier(random_state=settings.misc.seed,
                                     n_estimators=5,
                                     max_depth=3,
                                     n_jobs=-1)
        elif model_name == "KNN": 
            return KNeighborsClassifier(n_jobs=-1)        
        else:
            raise ValueError(f"{model_name} is not a valid estimator name.")


@pytest.fixture
def palette() -> Dict[str, str]:
    colors = [Colors.red, Colors.blue, Colors.yellow, Colors.violet, Colors.green2]

    return colors


@pytest.fixture
def rng() -> np.random.Generator:
    """Construct a new Random Generator using the seed specified in settings.

    Returns:
        numpy.random.Generator: a Random Generator based on BitGenerator(PCG64)
    """
    return np.random.default_rng(settings.misc.seed)


@pytest.fixture
def cv_config():
    ts = datetime.timestamp(datetime.now())

    config = {
        "N_REPEATS": 2,
        "N_SPLITS": 10,
        "SHUFFLE": True,
        "SCALE": False,
        "CALIBRATE": True,
        "CALIB_FRACTION": 0.15,
        "RESULTS_PATH": Path(__file__).resolve().parent / f"cv_results/cv_results_{ts}.pkl",
        "DUMP": bool
    }

    return config
        

@pytest.fixture
def models_names():
    return ["EBM", "LR", "SVC", "RF", "KNN"]


@pytest.fixture
def cv_results(load, load_path, models_names, cv_config) -> CVModellingOutput:
    """Fit and validate multiple binary models in a cross-validation scheme.

    Parameters
    ----------
    load: bool
        Whether we should load already existing cross-validation results.
    load_path: Path
        Path to already existing CV results.
    models_names: List[str]
        A list of model names to be cross-validated
    
    Returns:
        CVModellingOutput: the output of an internal cross-validation procedure.
    """
    if load:
        with open(load_path, "rb") as f_r:
            cv_results = pickle.load(f_r)

        return cv_results

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    from interpret.glassbox import ExplainableBoostingClassifier

    from utils import select_features, get_feature_selector

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    outer_cv = RepeatedStratifiedKFold(n_repeats=cv_config.get("N_REPEATS"),
                                       n_splits=cv_config.get("N_SPLITS"),
                                       random_state=settings.misc.seed)
    inner_cv = StratifiedKFold(n_splits=cv_config.get("N_SPLITS"),
                               shuffle=cv_config.get("SHUFFLE"),
                               random_state=settings.misc.seed)

    
    model_factory = BinaryModelFactory()
    cv_results = dict()

    for model_name in models_names:
        LOGGER.info(f"Processing model {model_name}\n")
    
        gts_train = []
        gts_val = []
        probas_train = []
        probas_val = []
        gts_train_conc = []
        gts_val_conc = []
        probas_train_conc = []
        probas_val_conc = []

        models = []
        errors = []
        correct = []
        features = []
            
        for i, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            Xtemp, ytemp = X.iloc[train_idx, :], y.iloc[train_idx]
            Xval, yval = X.iloc[val_idx, :], y.iloc[val_idx]

            if cv_config.get("CALIBRATE"):
                Xtrain, Xcal, ytrain, ycal = train_test_split(Xtemp, ytemp,
                                                              test_size=cv_config.get(
                                                                  "CALIB_FRACTION"),
                                                              stratify=ytemp,
                                                              random_state=settings.misc.seed)
            else:
                Xtrain, ytrain = Xtemp, ytemp

            model = model_factory.get_model(model_name)

            # select features
            feat_subset = select_features(Xtrain, ytrain,
                                          selector=get_feature_selector(
                                              settings.misc.seed),
                                          cv=inner_cv,
                                          scale=False,
                                          frac=1.25)
            features.append(feat_subset)

            if cv_config.get("SCALE"):
                numeric_cols = Xtrain.select_dtypes(
                    include=[np.float64, np.int64]).columns.tolist()
                scaler = StandardScaler()
                Xtrain.loc[:, numeric_cols] = scaler.fit_transform(
                    Xtrain.loc[:, numeric_cols])
                Xtest.loc[:, numeric_cols] = scaler.transform(
                    Xtest.loc[:, numeric_cols])

            model.fit(Xtrain.loc[:, feat_subset], ytrain)

            if cv_config.get("CALIBRATE"):
                model = get_calibrated_model(model,
                                             X=Xcal.loc[:, feat_subset],
                                             y=ycal)

            models.append(model)

            # accumulate ground-truths
            gts_train.append(ytrain)
            gts_val.append(yval)

            # accumulate predictions
            train_pred_probas = model.predict_proba(Xtrain.loc[:, feat_subset])[:, 1]
            val_pred_probas = model.predict_proba(Xval.loc[:, feat_subset])[:, 1]

            probas_train.append(train_pred_probas)
            probas_val.append(val_pred_probas)

            # identify correct and erroneous predictions according to the
            # classification cut-off that maximizes the Youden's J index
            fpr, tpr, thresholds = roc_curve(ytrain, train_pred_probas)
            idx = np.argmax(tpr - fpr)
            youden = thresholds[idx]

            labels_val = np.where(val_pred_probas >= youden, 1, 0)

            # indexes of validation instances misclassified by the model
            error_idxs = Xval[(yval != labels_val)].index
            errors.append(error_idxs)

            # indexes of correct predictions
            correct.append(Xval[(yval == labels_val)].index)

        # CV results for current model
        curr_est_results = CVModellingOutput(
            gts_train=np.array(gts_train),
            gts_val=np.array(gts_val),
            probas_train=np.array(probas_train),
            probas_val=np.array(probas_val),
            gts_train_conc=np.concatenate(gts_train),
            gts_val_conc=np.concatenate(gts_val),
            probas_train_conc=np.concatenate(probas_train),
            probas_val_conc=np.concatenate(probas_val),
            models=models,
            errors=np.array(errors),
            correct=np.array(correct),
            features=np.array(features)
        )
        
        cv_results[model_name] = curr_est_results

    cv_config.get("RESULTS_PATH").parent.mkdir(parents=True, exist_ok=True)

    if cv_config.get("DUMP"):
        with open(cv_config.get("RESULTS_PATH"), "wb") as f_w:
            pickle.dump(cv_results, f_w)

    return cv_results
