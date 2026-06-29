# K-Nearest Neighbors Classification

## Overview
This project walks through implementing K-Nearest Neighbors (KNN) from scratch in Python, then replicating the same workflow using scikit-learn. The primary dataset is the UCI Red Wine Quality dataset, where the goal is to classify wines as "good" or "bad" based on physicochemical measurements. A secondary analysis applies the same pipeline to a heart disease classification dataset.

## Biological / Data Science Question
Can we accurately classify red wine quality using only measurable chemical properties like acidity, sulfur dioxide levels, and alcohol content — without ever tasting the wine?

## Datasets
- **Red Wine Quality** — UCI Machine Learning Repository, accessed via Google Drive  
  Download: `!gdown 1RnmUfRSJWZxM7Ars3N3kTUEOBIJLSfrl`  
  Place the file `winequality-red.csv` in the root project directory before running.

- **Heart Disease** — UCI Heart Disease Dataset, accessed via Google Drive  
  Download: `!gdown 1A7dxaHxMog5rFWiNoORmeFE41cY6AFRT`  
  Place the file `heart.csv` in the root project directory before running.

Data files are not included in this repository. Both can be downloaded automatically by running the notebook cells.

## Methods & Tools

**Language:** Python 3.10+

**Libraries:**
| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | KNN classifier, preprocessing, evaluation |

**Pipeline:**
1. Load and explore the wine quality dataset
2. Create a binary label (good = quality ≥ 6)
3. Normalize features using z-score standardization
4. Implement Euclidean distance and KNN from scratch
5. Classify unknown wine samples
6. Replicate with scikit-learn using a proper train/test split
7. Evaluate with confusion matrix, accuracy, precision, recall, and F1
8. Sweep K values (1–30) to identify the optimal hyperparameter
9. Apply the complete pipeline to the heart disease dataset

## Key Results
- Baseline KNN (K=5) on the wine dataset achieves ~70–72% accuracy on the test set
- Optimal K varies by run but typically falls in the range of K=10–20
- The from-scratch implementation produces predictions consistent with the scikit-learn version, confirming correctness
- The same pipeline generalizes well to the heart disease dataset with minimal changes

## How to Run

**Option 1 — Google Colab (recommended)**
1. Upload `KNN_Classification.ipynb` to Google Colab
2. Run all cells top to bottom — the datasets download automatically

**Option 2 — Local environment**
```bash
git clone https://github.com/yourusername/knn-wine-classification
cd knn-wine-classification
pip install -r requirements.txt
jupyter notebook KNN_Classification.ipynb
```
Then manually download the datasets using the `gdown` cells or the links above.

## Project Structure
```
knn-wine-classification/
├── README.md
├── KNN_Classification.ipynb   # Main notebook
├── requirements.txt
├── data/                      # Not tracked — see Data section above
└── figures/                   # Output plots (optional)
```
