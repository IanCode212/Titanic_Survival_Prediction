# Titanic Survival Prediction

Who survived the Titanic, and why? This is my take on the classic Kaggle problem, built for DMA Lab 5. The short version: **good features did most of the work, and the choice of model barely mattered.**

## The question

Given a passenger's ticket class, sex, age, family and ticket details, predict whether they survived. The training set has 891 passengers (38% survived), and the test set has 418 passengers whose outcomes are scored on Kaggle.

## How I approached it

**1. Look at the data first.** Sex and class split survival more than anything else. Nearly every 1st and 2nd-class woman survived, about half of 3rd-class women did, and men had low survival in every class. The title in each name (Mr, Mrs, Miss, Master) turned out to carry age and family information that the raw columns hide.

**2. Set a baseline.** Predicting that everyone died gives 61.6%. A hand-written rule (women survive, except 3rd-class women who boarded at Southampton) scored **0.778** on Kaggle. Every model after that had to beat it.

**3. Engineer features, with a reason for each one.**

| Feature group | Why |
|---|---|
| Sex × Class (6 flags) | Sex and class interact, so give the model each combination directly |
| Family size (`Alone`, `Large_Family`) | Small families did best, and people alone or in large families did worse |
| Travel group (`Group_Size`) | People on the same ticket travelled together, even when they weren't related |
| Title (Mr / Mrs / Miss / Master / Rare) | Captures age and status; "Master" marks young boys |
| Child (Age < 13 or Master) | Candidate feature; later removed as redundant |

I dropped Cabin (mostly missing), Embarked (little signal once class is known), Fare (repeats class and group size) and raw Age (20% missing).

**4. Check the features before trusting them.** A correlation heatmap showed a few near-copies, for example `Fam_Size` vs `Group_Size` (r = 0.82) and `Child` vs `Title:Master` (r = 0.73). A drop-one check then removed features one at a time and kept a removal only if cross-validation didn't get worse. It removed `Fam_Size`, `Non_Family_Group` and `Child`, leaving 14 features.

**5. Compare models honestly.** I used 10×5 repeated stratified cross-validation (50 fits per score), because a single 5-fold split moves by about a point from run to run. I always tracked the train/CV gap to catch overfitting.

**6. Try a family-survival idea.** Families often lived or died together, so I matched families by surname + ticket and used how other members fared as a feature, with one logistic regression per sex × class group. These rates come from training labels, so they're rebuilt inside every CV fold, and a passenger's own label is never used.

## Results (10×5-fold CV)

**Adding features, one group at a time** (logistic regression):

| Features | CV accuracy |
|---|---|
| Sex × Class | 77.3% |
| + Family size | 80.0% |
| + Travel group | 79.9% |
| + Title | 83.2% |

**Comparing models on the same 14 features:**

| Model | CV accuracy |
|---|---|
| Per-group logistic regression + family survival rates | **83.9%** |
| Linear regression (0.5 cutoff) | 83.2% |
| Ridge regression (0.5 cutoff) | 83.2% |
| Logistic regression (L2) | 83.2% |
| L1 logistic regression with interaction terms (**submitted**) | 83.0% |
| Gradient boosting (depth 2) | 83.0% |

Fold-to-fold spread is about ±2.4 points, so everything from 83.0% to 83.2% is a tie. I submitted the L1 interaction model: it's simple, keeps only 20 of its 105 interaction columns, and doesn't depend on other passengers' labels.

## What I learned

- **Features beat models.** Going from one feature group to four added about 6 points. Switching between six different models moved accuracy by less than 1 point.
- **More features aren't always better.** Redundant features didn't improve CV, and some widened the train/CV gap.
- **Use repeated CV for close calls.** Single 5-fold scores swung by up to a point between runs, which is enough to pick the wrong model.
- **CV and the leaderboard can disagree.** An earlier version with Title improved CV but dropped the public score from 0.778 to 0.760. The public leaderboard is only a small slice of the test set, so I trust CV more.

## Repository

| File | What it is |
|---|---|
| `titanic.ipynb` | The full analysis, from exploration to submission, with outputs |
| `ian_prediction.csv` | Rule-based baseline submission |
| `ian_final_prediction.csv` | Final model submission |
| `requirements.txt` | Python packages |

## Run it

1. Download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) into a `data/` folder.
2. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run `titanic.ipynb` top to bottom. The repeated-CV cells take a few minutes.
