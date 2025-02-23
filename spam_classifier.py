import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Kaggle dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham':0,'spam':1})

# Preprocess with TF-IDF
tf_idf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tf_idf.fit_transform(df['text'])
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_perd = model.predict(X_test)
accuracy = accuracy_score(y_test, y_perd)
report = classification_report(y_test, y_perd, target_names=['Ham', 'Spam'])
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Classify new documents
new_documents = [
    "Win a free iPhone now!",
    "Hey, how are you today?"
]
new_vectors = tf_idf.transform(new_documents)
predictions = model.predict(new_vectors)

for doc, pred in zip(new_documents, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Document: '{doc}' | Predicted: {label}")