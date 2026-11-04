# Titanic Survival Prediction

This is my project for DMA Lab 5, based on the Kaggle Titanic competition. The goal is to predict which passengers survived. My final model scored 0.80478 on the private leaderboard. The thing that got it past 0.80 wasn't a fancier model but one extra piece of information: what happened to the rest of each passenger's family.

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
I used 10x5 repeated stratified cross-validation (50 fits per score), because one 5-fold split can move by about a point between runs. I also tracked the gap between train and CV accuracy to catch overfitting. With the 15 features, every model I tried (linear, ridge, logistic, boosting, L1 logistic regression with interaction terms) landed between 82.9% and 83.2%. The best one scored 0.79282 on Kaggle.

### 6. Adding family survival
Families often lived or died together. I matched families by surname and ticket, and for each passenger I worked out how the other family members in the training data did. These rates come from the training labels, so I rebuild them inside every CV fold and never use a passenger's own label. Otherwise the CV score would be too optimistic.

I added the family rates to ten different models. I also tested a simple women-child group rule:
- Men die.
- Women survive, unless every other woman or boy in their family died.
- Boys (title Master) die, unless every other woman or boy in their family survived.

## Results

Models with family survival rates (10x5 CV):

| Model | CV accuracy |
|---|---|
| Vote: SVM + women-child rule + gradient boosting | 84.3% |
| SVM (RBF) + family | 84.1% |
| Women-child group rule | 83.7% |
| Gradient boosting + family | 83.7% |
| Logistic regression + family | 83.6% |
| L1 interactions + family | 83.6% |
| KNN + family | 83.3% |
| Random forest + family | 83.3% |
| L1 interactions, no family | 83.2% |
| Extra trees + family | 83.2% |

Kaggle scores:

| Submission | Private | Public |
|---|---|---|
| Rule-based baseline | | 0.778 |
| L1 interactions, no family | 0.79282 | 0.76047 |
| SVM + family | 0.80079 | 0.76047 |
| Vote: SVM + rule + gradient boosting | 0.80478 | 0.77245 |
| Women-child group rule (final) | 0.80478 | 0.77844 |

I picked the women-child group rule as the final model. It ties for the best private score, has the best public score, and it doesn't need any training.

## What I learned

- Features matter more than the model. Going from one feature group to four added about 6 points, while switching between models changed less than half a point.
- The right information beats a better algorithm. The family survival rates were what pushed every model past 0.80, and a three-line rule using them did as well as the trained models.
- A small CV gain can just be noise. The vote had the best CV but didn't beat the simple rule on Kaggle, and removing overlapping features looked better in CV but scored worse.
- Features built from labels can leak. The family survival rates are only fair if they are rebuilt inside each fold from the training labels.

## Files

| File | Description |
|---|---|
| titanic.ipynb | The full notebook, from exploration to submission |
| ian_prediction.csv | Rule-based baseline submission |
| ian_final_prediction.csv | Final submission (women-child group rule) |
| ian_submission_vote.csv | Candidate: vote of SVM, rule and gradient boosting |
| ian_submission_svm_family.csv | Candidate: SVM with family features |
| ian_submission_wcg_rule.csv | Candidate: women-child group rule (same as the final) |
| requirements.txt | Python packages |

## How to run

1. Download train.csv and test.csv from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) into a folder called data.
2. Install the packages with pip install -r requirements.txt.
3. Run titanic.ipynb from top to bottom. The repeated CV cells take a few minutes.
