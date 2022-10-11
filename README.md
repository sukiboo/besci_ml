# BeSci + ML
An ambitious project of combining machine learning and behavioral science.
Currently it consists of several disjoint pieces of code split into the following folders:

1. **coin_flips** simulates the data of flipping coins described by dummy features and analyzes the model trained on that data.
In particular, it shows a problem of using **shap** on one-hot encoded data.

2. **nhanes_dbq_synthesis** uses NHANES DBQ-dataset and builds a simulator using additional behavioral rules.
In this formlulation the input consist of 6 demographic features: age, gender, race, income, education, marital status.
The output is a 5-dimensional probability vector of food preferences.
The resulting simulator is non-deterministic and synthesizes data that alignes with the underlying behavioral rules.

3. **nhanes_dbq_explanation** uses 6 years of NHANES DBQ-surveys to learn people's food preferences from 6 demographic features and then analyze their importance.


