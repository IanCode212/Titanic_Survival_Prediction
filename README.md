# Titanic Survival Prediction

Course project (DMA Lab 5, Kaggle in-class competition): predict which Titanic passengers survived.

## Approach
1. **Exploration:** survival rates by sex, class, port and age.
2. **Rule-based classifier (5a):** women survive, except 3rd-class women from Southampton. Public score 0.77844.
3. **Models (5b):** Decision Tree, Logistic Regression and a Neural Network, compared with 5-fold cross-validation,
   tracking the train/CV gap to watch for overfitting.
4. **Ensembles:** soft voting, bagging, random forest, AdaBoost and stacking.
5. **Feature engineering:** family size; Title from Name (rare titles grouped); missing ages filled by Title.

## Results
| Model | CV accuracy | Gap |

| Stacking (with Title) | 0.827 | 0.020 |
| Bagging | 0.833 | 0.064 |
| Logistic Regression | 0.825 | 0.007 |

Final: Stacking. Adding Title improved CV but lowered the public score (0.778 → 0.760), a reminder that a
167-passenger public leaderboard is noisy.

## Run it
1. Download `train.csv`, `test.csv` and `gender_submission.csv` from the Kaggle competition into `data/`
2. `pip install -r requirements.txt`
3. Run `titanic.ipynb` top to bottom