# Changelog

<!--next-version-placeholder-->

## v0.2.0 (17/07/2023)

### Feature

- Introduced the curves module for producing average receiver-operating characteristic curved after cross-validation.
- Added a custom dataclass type for the output of a cross-validation procedure: `CVModellingOutput`.

### Tests

- Added a fixture to simulate the output of a cross-validation procedure and added tests for the average receiver-operating characteristic curves.

## v0.1.0 (23/06/2023)

- First release of `modelsight`!
- Calibration module to assess calibration of ML predicted probabilities via Hosmer-Lemeshow plot.