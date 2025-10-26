
import pandas as pd
from datasets import load_dataset

ds = load_dataset("google/civil_comments")

# ------------ DATA UPLOAD & PRE PROCESSING -------------
# Upload data
jigsaw = pd.read_csv('data/jigsaw.csv')[["comment_text", 'toxic']]
noisyHate = pd.read_csv("hf://datasets/NoisyHate/Noisy_Hate_Data/toxic detection model test set with perturbations.csv")[['perturbed_version', 'toxicity']]
youtube = pd.read_csv('data/youtoxic_english_1000.csv')[["Text", 'IsToxic']]
toxic_comments = pd.read_csv('data/toxicity_en.csv')[['text', 'is_toxic']]

# Process jigsaw toxic comments data
jigsaw = jigsaw.rename(columns={'comment_text': 'text', 'toxic': 'isToxic'})
jigsaw = jigsaw.dropna(subset=['text'])
jigsaw = jigsaw.drop_duplicates(subset=['text'])
jigsaw['isToxic'] = jigsaw['isToxic'].apply(lambda x: 'toxic' if x==1 else 'safe')

# Process noisyHate data
noisyHate = noisyHate.rename(columns={'perturbed_version': 'text', 'toxicity': 'isToxic'})
noisyHate = noisyHate.dropna(subset=['text'])
noisyHate = noisyHate.drop_duplicates(subset=['text'])
noisyHate['isToxic'] = noisyHate['isToxic'].apply(lambda x: 'toxic' if x<0.51 else 'safe')

# Process YouTube comments data
youtube = youtube.rename(columns={'is_toxic': 'isToxic'})
youtube = youtube.dropna(subset=['Text'])
youtube = youtube.drop_duplicates(subset=['Text'])
youtube['isToxic'] = noisyHate['isToxic'].apply(lambda x: 'toxic' if x == 'TRUE' else 'safe')

# Process toxicity dataset
toxic_comments = toxic_comments.rename(columns={'perturbed_version': 'text', 'is_toxic': 'isToxic'})
toxic_comments = toxic_comments.dropna(subset=['text'])
toxic_comments = toxic_comments.drop_duplicates(subset=['text'])
toxic_comments['isToxic'] = toxic_comments['isToxic'].apply(lambda x: 'toxic' if x == 'Toxic' else 'safe')

# Combine into one dataset
all_data = pd.concat([jigsaw[['text', 'isToxic']], jigsaw[['text', 'isToxic']], jigsaw[['text', 'isToxic']], toxic_comments], ignore_index=True)

# balance dataset with safe and toxic messages
from sklearn.utils import resample

# Separate majority and minority classes
df_safe = all_data[all_data['isToxic'] == 'safe']
df_toxic = all_data[all_data['isToxic'] == 'toxic']

# Downsample safe class to match toxic count
df_safe_balanced = resample(df_safe, 
                            replace=False,        # sample without replacement
                            n_samples=len(df_toxic), # match toxic size
                            random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_safe_balanced, df_toxic])

# Shuffle rows
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save if you want
df_balanced.to_csv("balanced_dataset.csv", index=False)

# check balance
#print(df_balanced['isToxic'].value_counts())

# check for any null values, none
#print(df_balanced['isToxic'].isnull().value_counts())

# Split data
from sklearn.model_selection import train_test_split

# Split into train/test
X = df_balanced['text']
y = df_balanced['isToxic'].map({'safe': 0, 'toxic': 1})  # binary labels

# 20% of data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import numpy as np

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------- TF IDF + Logistic Regression ----------------

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
print("Logistic Regression:\n", classification_report(y_test, lr_preds))

# Log reg confusion matrix
cm = confusion_matrix(y_test, lr_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "toxic"])
disp.plot(cmap=plt.cm.Blues)

# Measure inference time 

# Example: Measure time for 100 samples
samples = X_test_tfidf[:100]

start = time.time()
predictions = lr_model.predict(samples)  # for TF-IDF based
end = time.time()

avg_inference_time = (end - start) / 100
print(f"LR Avg inference time: {avg_inference_time * 1000:.5f} ms per message")

# ------------------- TF IDF + Naive Bayes ------------------

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
print("Naive Bayes:\n", classification_report(y_test, nb_preds))

# NB Confusion matrix
cm = confusion_matrix(y_test, nb_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "toxic"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Measure inference time 

# Example: Measure time for 100 samples
samples = X_test_tfidf[:100]

start = time.time()
predictions = nb_model.predict(samples)  # for TF-IDF based
end = time.time()

avg_inference_time = (end - start) / 100
print(f"NB Avg inference time: {avg_inference_time * 1000:.5f} ms per message")

# TRANSFORMERS
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

# Load model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
