# Modelsight

Better insights into Machine Learning models performance.

Modelsight is a collection of functions that create publication-ready graphics for the visual evaluation of binary classifiers adhering to the scikit-learn interface. 

Modelsight is strongly oriented towards the evaluation of already fitted `ExplainableBoostingClassifier` of the [interpretml](https://github.com/interpretml/interpret) package.

## Installation
```console
$ pip install modelsight
```

## Usage
See the [example](/docs/example.ipynb) notebook. 

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`modelsight` was created by Francesco Pisu. It is licensed under the terms of the GNU General Public License v3.0 license.

## Roadmap
Features:
- [x] Average Receiver-operating characteristic curves
- [ ] Average Precision-recall curves
- [ ] Feature importance (Global explanation)
- [ ] Individualized explanations (Local explanation)

CI/CD:
- [ ] Integration with GH Actions

## Credits

`modelsight` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
