# 📧 Spam Mail Detection with Logistic Regression

This project is a **Spam Mail Classifier** built using **Logistic Regression** and **TF-IDF Vectorization**. It is trained on a dataset of labeled email messages to distinguish between **spam** and **ham (non-spam)** emails.

## 🔍 Features

- Classifies an email as **Spam** or **Ham**
- Uses **TF-IDF** for feature extraction
- Implements **Logistic Regression** as the ML algorithm
- Achieves good accuracy on both training and testing datasets
- Includes functionality to test your own email input

## 🧠 Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

## 📁 Dataset

The project uses a CSV file named `mail_data.csv`, which contains labeled email messages. Each message is tagged as **ham (1)** or **spam (0)**.

Example structure of the CSV:
| Category | Message |
|----------|---------|
| ham      | Hello, how are you? |
| spam     | Congratulations! You've won a free iPhone! |

## 🚀 How It Works

1. **Load and preprocess the data**
2. **Convert the text messages into numerical features** using `TfidfVectorizer`
3. **Split the dataset** into training and testing data
4. **Train the Logistic Regression model**
5. **Predict and evaluate** the model on both training and test data
6. **Allow user input** to check if a custom message is spam or ham

## 📈 Accuracy

- **Training Accuracy**: ~98%
- **Testing Accuracy**: ~96%

## 🧪 Example Usage

```python
input_data = """Please note that product prices and availability are subject to change."""
input_data_features = vectorizer.transform([input_data])
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("This is a HAM mail")
else:
    print("This is a SPAM mail")
