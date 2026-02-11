import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "email": [
        "Win a free iPhone now click here",
        "Your bank account is compromised login now",
        "Congratulations you won a lottery",
        "Meeting scheduled at 10am tomorrow",
        "Please review the project document",
        "Let's have lunch today",
        "Claim your free prize now",
        "Update your password immediately"
    ],
    "label": [
        "phishing","phishing","phishing",
        "safe","safe","safe",
        "phishing","phishing"
    ]
}

df = pd.DataFrame(data)

df["label"] = df["label"].map({"safe":0,"phishing":1})

X_train,X_test,y_train,y_test = train_test_split(
    df["email"],df["label"],test_size=0.3,random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

print("\nAccuracy:",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

print("\n--- Test Your Own Email ---")
user_email = input("Enter email text: ")

user_vec = vectorizer.transform([user_email])
prediction = model.predict(user_vec)

if prediction[0] == 1:
    print("⚠️ This email is PHISHING")
else:
    print("✅ This email is SAFE")
