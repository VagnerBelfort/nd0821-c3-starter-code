# Model Card


## Model Details
- Scikit-Learn GradientBoosting algorithm
- Optimized for a the binary classification of salaries (>50K, <=50K)

## Intended
- MLOps pipeline using github action, DVC and Heroku / FastAPI deployment
## Training Data
- Census Income Data Set: https://archive.ics.uci.edu/ml/datasets/census+income
```
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- salary: >50K, <=50K.
```
## Evaluation Data
- Census Income Data Set - used cross validation to train the model on the entire census dataset
## Metrics
- Evaluation metrics include:
    - roc_auc score
    - f1 score
    - precision
    - recall

- Best model performance:
    - roc_auc = 0.94
    - precision = 0.81
    - recall = 0.68
    - fbeta(1) = 0.74

## Ethical Considerations
- Uses all features from the census dataset including race and sex

- marital status is the most important feature
## Caveats and Recommendations
- The dataset is somewhat imbalanced with approximately 25% of labels >50K and 75%% <=650K