import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

file_paths = [
    r"D:\codesoft dataset\task 1\description.txt.txt",
    r"D:\codesoft dataset\task 1\test_data.txt",
    r"D:\codesoft dataset\task 1\test_data_solution.txt",
    r"D:\codesoft dataset\task 1\train_data.txt"
]

texts, labels = [], []
for filepath in file_paths:
    genre = os.path.splitext(os.path.basename(filepath))[0]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(genre)

# Remove rare labels
counter = Counter(labels)
texts = [t for t, l in zip(texts, labels) if counter[l] > 10]
labels = [l for l in labels if counter[l] > 10]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
