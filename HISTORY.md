# Release History

## dev

### Improvements

## 0.1.3 (2019-04-27)

### Improvements

- Delete models if creation or deployment fails
- Add model metadata with scikit-learn version
- Auto-refresh of models when being modified
- Prevent re-using deleted models

## 0.1.2 (2019-04-19)

### Improvements

- Allow to set host and api key through environment variables
- Add support for ModelVersions and updating models
- Prevent from deploying untrained models and custom estimators
- Nicer objects **repr**

## 0.1.1 (2019-04-14)

### Bug Fixes

- Include numpy as dependency

## 0.1.0 (2019-04-14)

### Features

- Initial Release with support for Scikit Learn models and pipeline
  - Deploy models
  - Get single predictions from deployed models
  - Get batch predictions from deployed models
  - Rename models
  - Delete models
