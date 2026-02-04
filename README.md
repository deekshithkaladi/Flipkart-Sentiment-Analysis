# Flipkart Sentiment Analysis Project

## Objective
Classify customer reviews of YONEX MAVIS 350 Shuttle into Positive or Negative using Machine Learning and Optuna Hyperparameter Tuning.

---

## Dataset
- 8,518 Flipkart product reviews
- Features: Review Text, Rating, Title, Votes, Location

---

## Tech Stack
- Python
- Scikit-learn
- Optuna
- TF-IDF
- Streamlit
- AWS EC2

---

## Model Details
- Text Cleaning with NLTK
- TF-IDF Vectorization
- Logistic Regression
- Hyperparameter tuning using Optuna
- Evaluation Metric: F1 Score

---

## Results
- Accuracy: ~91%
- Optimized F1 Score
- Successfully deployed using Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
