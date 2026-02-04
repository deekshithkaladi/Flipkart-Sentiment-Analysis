import pandas as pd
df1 = pd.read_csv("data/data.csv")
df2 = pd.read_csv("data/data2.csv")
df3 = pd.read_csv("data/data3.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Total Rows:", df.shape)
print(df.head())
def label_sentiment(rating):
    if rating >= 4:
        return 1
    else:
        return 0

df['sentiment'] = df['Ratings'].apply(label_sentiment)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['Review text'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
def objective(trial):

    # TF-IDF tuning
    max_features = trial.suggest_int("max_features", 2000, 10000)
    ngram_range = trial.suggest_categorical("ngram_range", [(1,1), (1,2)])

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    X = tfidf.fit_transform(df['clean_review'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression tuning
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000
    )

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="f1"
    ).mean()

    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Parameters:", study.best_trial.params)
best_params = study.best_trial.params

tfidf = TfidfVectorizer(
    max_features=best_params["max_features"],
    ngram_range=best_params["ngram_range"]
)

X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(
    C=best_params["C"],
    solver=best_params["solver"],
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, f1_score

print(classification_report(y_test, y_pred))
print("Final F1 Score:", f1_score(y_test, y_pred))
import pickle

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model saved successfully!")
