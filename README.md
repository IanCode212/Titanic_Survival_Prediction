# Titanic Survival Prediction

This is my project for DMA Lab 5, based on the Kaggle Titanic competition. The goal is to predict which passengers survived. What I found is that the features made a much bigger difference than the choice of model.

## Data

The training set has 891 passengers, and 38% of them survived. The test set has 418 passengers, and Kaggle scores the predictions for them. Each passenger has a ticket class, sex, age, family counts, ticket number, fare, cabin and port.

## What I did

### 1. Exploring the data
Sex and class split survival more than anything else. Almost all women in 1st and 2nd class survived, about half of the women in 3rd class did, and men had low survival in every class. The title in the name (Mr, Mrs, Miss, Master) also turned out to be useful, because it tells you about age and family.

### 2. Baseline
If you predict that everyone died, you get 61.6%. My hand-written rule (women survive, except 3rd class women who boarded at Southampton) got 0.778 on Kaggle, so every model had to beat that.

### 3. Features
| Feature group | Why I used it |
|---|---|
| Sex and class (6 flags) | Sex and class work together, so the model gets each combination directly |
| Family size, Alone, Large_Family | Small families did best, people alone or in big families did worse |
| Group_Size, Non_Family_Group | People on the same ticket travelled together, even when they weren't family |
| Title (Mr, Mrs, Miss, Master) | Gives age and status, and Master means a young boy |

I didn't use Cabin (mostly missing), Embarked (not much signal once class is known), Fare (repeats class and group size) or Age (20% missing).

### 4. Checking the features
A correlation heatmap shows that the family features overlap a lot. For example, Fam_Size and Group_Size have a correlation of 0.82. I tried removing the overlapping ones and adding a Child flag. CV only went up by 0.2 points and the Kaggle score went down, so I kept the original 15 features.

### 5. Comparing models
I used 10x5 repeated stratified cross-validation (50 fits per score), because one 5-fold split can move by about a point between runs. I also tracked the gap between train and CV accuracy to catch overfitting.

### 6. Family survival experiment
Families often lived or died together. I matched families by surname and ticket, used how the other family members did as a feature, and trained one logistic regression for each sex and class group. The family rates come from the training labels, so I rebuild them inside every CV fold and never use a passenger's own label.

## Results

Adding one feature group at a time (10x5 CV):

| Features | Logistic regression | Random forest |
|---|---|---|
| Sex and class | 77.3% | 77.3% |
| + Family size | 80.0% | 80.4% |
| + Travel group | 79.5% | 80.4% |
| + Title | 82.9% | 83.2% |

Different models on the same 15 features (10x5 CV):

| Model | CV accuracy |
|---|---|
| Per-group logistic regression + family survival rates | 83.6% |
| L1 logistic regression with interaction terms (submitted) | 83.2% |
| Ridge regression (0.5 cutoff) | 83.1% |
| Linear regression (0.5 cutoff) | 83.0% |
| Gradient boosting (depth 2) | 83.0% |
| Logistic regression (L2) | 82.9% |

The scores change by about 2.4 points from fold to fold, so these models are all very close. I submitted the L1 interaction model because it has no gap between train and CV, only keeps 21 of its 120 columns, and doesn't depend on other passengers' labels.

Kaggle scores:

| Submission | Private | Public |
|---|---|---|
| Rule-based baseline | | 0.778 |
| L1 interaction model, 15 features (final) | 0.79282 | 0.76047 |
| Same model, overlapping features removed and Child added | 0.78884 | 0.75449 |

## What I learned

- Features matter more than the model. Going from one feature group to four added about 6 points, while switching between six models changed less than half a point.
- A small CV gain can just be noise. Removing the overlapping features looked 0.2 points better in CV but scored worse on Kaggle.
- Repeated CV helps when scores are close. A single 5-fold split moved by up to a point between runs, which is enough to pick the wrong model.
- Features built from labels can leak. The family survival rates are only fair if they are rebuilt inside each fold from the training labels.

## Files

| File | Description |
|---|---|
| titanic.ipynb | The full notebook, from exploration to submission |
| ian_prediction.csv | Rule-based baseline submission |
| ian_final_prediction.csv | Final model submission |
| requirements.txt | Python packages |

## How to run

1. Download train.csv and test.csv from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) into a folder called data.
2. Install the packages with pip install -r requirements.txt.
3. Run titanic.ipynb from top to bottom. The repeated CV cells take a few minutes.
