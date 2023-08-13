<img src="https://github.com/francescopisu/modelsight/blob/main/docs/images/modelsight-logo.svg" width="175" align="right" />

# Welcome to modelsight

> Better insights into Machine Learning models performance.

Modelsight is a collection of functions that create publication-ready graphics for the visual evaluation of binary classifiers adhering to the scikit-learn interface. 

Modelsight is strongly oriented towards the evaluation of already fitted `ExplainableBoostingClassifier` of the [interpretml](https://github.com/interpretml/interpret) package.

| Overview | |
|---|---|
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/modelsight?color=orange)](https://pypi.org/project/modelsight/) [![!python-versions](https://img.shields.io/pypi/pyversions/modelsight)](https://www.python.org/) |

## 💫 Features
Our goal is to streamline the visual assessment of binary classifiers by creating a set of functions designed to generate publication-ready plots. 


| Module | Status | Links |
|---|---|---|
| **[Calibration]** | maturing | [Tutorial](https://modelsight.readthedocs.io/en/latest/01_calibration.html) · [API Reference](https://modelsight.readthedocs.io/en/latest/autoapi/modelsight/calibration/index.html) |
| **[Curves]** | maturing | [Tutorial](https://modelsight.readthedocs.io/en/latest/02_curves.html) · [API Reference](https://modelsight.readthedocs.io/en/latest/autoapi/modelsight/curves/index.html) |

[calibration]: https://github.com/francescopisu/modelsight/tree/main/src/modelsight/calibration
[curves]: https://github.com/francescopisu/modelsight/tree/main/src/modelsight/curves

## :eyeglasses: Install modelsight
- **Operating system**: macOS X · Linux
- **Python version**: 3.10 (only 64-bit)
- **Package managers**: [pip]

[pip]: https://pip.pypa.io/en/stable/

### pip
Using pip, modelsight releases are available as source packages and binary wheels. You can see all available wheels [here](https://pypi.org/simple/modelsight/).
```console
$ pip install modelsight
```

## :zap: Quickstart
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from interpret.glassbox import ExplainableBoostingClassifier

from utils import (
    select_features, 
    get_feature_selector, 
    get_model, 
    get_calibrated_model
)

from modelsight.curves import average_roc_curves
from modelsight.config import settings
from modelsight._typing import CVModellingOutput, Estimator

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

outer_cv = RepeatedStratifiedKFold(n_repeats=cv_config.get("N_REPEATS"), 
                        n_splits=cv_config.get("N_SPLITS"), 
                        random_state=settings.misc.seed)
inner_cv = StratifiedKFold(n_splits=cv_config.get("N_SPLITS"), 
                        shuffle=cv_config.get("SHUFFLE"),
                        random_state=settings.misc.seed)

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

ts = datetime.timestamp(datetime.now())
cv_config = {
    "N_REPEATS": 10,
    "N_SPLITS": 10,
    "SHUFFLE": True,
    "SCALE": False,
    "CALIBRATE": True,
    "CALIB_FRACTION": 0.15,
}

for i, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
    Xtemp, ytemp = X.iloc[train_idx, :], y.iloc[train_idx]
    Xval, yval = X.iloc[val_idx, :], y.iloc[val_idx]

    if cv_config.get("CALIBRATE"):
        Xtrain, Xcal, ytrain, ycal = train_test_split(Xtemp, ytemp, 
                                test_size=cv_config.get("CALIB_FRACTION"), 
                                stratify=ytemp, 
                                random_state=settings.misc.seed)
    else:
        Xtrain, ytrain = Xtemp, ytemp
    
    model = get_model(seed=settings.misc.seed)

    # select features
    feat_subset = select_features(Xtrain, ytrain, 
                                selector=get_feature_selector(settings.misc.seed), 
                                cv=inner_cv,
                                scale=False,
                                frac=1.25)
    features.append(feat_subset)

    if cv_config.get("SCALE"):
        numeric_cols = Xtrain.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
        scaler = StandardScaler()
        Xtrain.loc[:, numeric_cols] = scaler.fit_transform(Xtrain.loc[:, numeric_cols])
        Xtest.loc[:, numeric_cols] = scaler.transform(Xtest.loc[:, numeric_cols])            

    model.fit(Xtrain[feat_subset], ytrain)

    if cv_config.get("CALIBRATE"):
        model = get_calibrated_model(model, 
                                    X=Xcal.loc[:, feat_subset],
                                    y=ycal)
    
    models.append(model)

    # accumulate ground-truths
    gts_train.append(ytrain)
    gts_val.append(yval)

    # accumulate predictions
    train_pred_probas = model.predict_proba(Xtrain)[:, 1]
    val_pred_probas = model.predict_proba(Xval)[:, 1]

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
    
    
cv_results = CVModellingOutput(
    gts_train = np.array(gts_train),
    gts_val = np.array(gts_val),
    probas_train = np.array(probas_train),
    probas_val = np.array(probas_val),
    gts_train_conc = np.concatenate(gts_train),
    gts_val_conc = np.concatenate(gts_val),
    probas_train_conc = np.concatenate(probas_train),
    probas_val_conc = np.concatenate(probas_val),
    models = models,
    errors = np.array(errors),
    correct = np.array(correct),
    features = np.array(features)
)

# Plot the average receiver-operating characteristic (ROC) curve
model_names = ["EBM"]
alpha = 0.05
alph_str = str(alpha).split(".")[1]
alpha_formatted = f".{alph_str}"
roc_symbol = "*"
n_boot = 100
kwargs = dict()

f, ax = plt.subplots(1, 1, figsize=(8, 8))

f, ax, barplot, bars, aucs_cis = average_roc_curves(cv_results,
                                                    colors=palette,
                                                    model_keys=model_names,
                                                    show_ci=True,
                                                    n_boot=n_boot,
                                                    bars_pos=[
                                                        0.5, 0.01, 0.4, 0.1*len(model_names)],
                                                    random_state=settings.misc.seed,
                                                    ax=ax,
                                                    **kwargs)
```

## :gift_heart: Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## 🛣️ Roadmap
Features:
- [x] Calibration module to assess calibration of ML predicted probabilities via Hosmer-Lemeshow plot
- [x] Average Receiver-operating characteristic curves
- [ ] Average Precision-recall curves
- [ ] Feature importance (Global explanation)
- [ ] Individualized explanations (Local explanation)
