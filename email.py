import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'email': [
        "Win a free iPhone now!", 
        "Hello, how are you doing today?", 
        "Exclusive offer just for you", 
        "Meeting at 5 PM, see you there", 
        "Congratulations! You have won a lottery",
        "Reminder: Your bank account needs verification"
    ],
    'label': [1, 0, 1, 0, 1, 1]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Create a text classification pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to predict new emails
def predict_spam(email_text):
    prediction = pipeline.predict([email_text])
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
new_email = "Get a free trip to Paris now!"
print(f'Email: "{new_email}" is {predict_spam(new_email)}')
