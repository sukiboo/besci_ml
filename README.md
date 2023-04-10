# BeSci + ML
An ambitious project of combining machine learning and behavioral science --- **Behavioral Machine Learning** (not sure what it is, we're trying to invent it).
Currently this repo consists of several disjoint pieces of code split into the following folders:

1. `besci_loss` compares the conventional ML to the *behavioral* ML on [NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/) DBQ-dataset. This whole project kinda failed because conventional ML performs better and I'm not sure what to do about it.

2. `coin_flips` simulates the data of flipping coins described by dummy features and analyzes the model trained on that data.
In particular, it shows a problem of using `shap` on one-hot encoded data.

3. `nhanes_dbq_synthesis` uses [NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/) DBQ-dataset and builds a simulator using additional behavioral rules.
In this formlulation the input consist of 6 demographic features: age, gender, race, income, education, marital status.
The output is a 5-dimensional probability vector of food preferences.
The resulting simulator is non-deterministic and synthesizes data that alignes with the underlying behavioral rules.

4. `nhanes_dbq_explanation` uses 12 years of [NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/) DBQ-surveys to learn people's food preferences from 6 demographic features and then analyze importance of these demographic features on the food choice preferences.

5. `brfss` contains the data from [BRFSS](https://www.cdc.gov/brfss/index.html) dataset, which has 450,000 rows and 358 columns. No idea what to do with all this data but it's here for future inspiration.
