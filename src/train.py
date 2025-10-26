import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing import preprocess_text
# from imblearn.over_sampling import SMOTE

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertTokenizer,  DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification


# from models import get_log_reg, get_naive_bayes, get_bert_model

# # --- Load unified dataset ---
# def load_dataset(sample_size=None):
#     df = pd.read_csv("toxic-spam-detection/data/unified_dataset.csv")

#     # Optional: downsample for BERT training
#     if sample_size:
#         df = df.sample(n=sample_size, random_state=42)

#     df["text"] = df["text"].astype(str).apply(preprocess_text)
#     X = df['text']
#     y = df['label']

#     # Stratified split
#     return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# # Split into train/test
# X_train, X_test, y_train, y_test = load_dataset()
# train_texts_small, test_texts_small, train_labels_small, test_labels_small = load_dataset(sample_size=5000)

# # --- TF-IDF representation ---
# vectorizer = TfidfVectorizer(
#     lowercase=True,
#     stop_words="english",
#     ngram_range=(1,2),  # unigrams + bigrams
#     max_features=20000
# )

# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # SMOTE 
# # use_smote = True
# # if use_smote:
# #     sm = SMOTE(random_state=42)
# #     X_train_vec, y_train = sm.fit_resample(X_train_vec, y_train)
# #     print("Applied SMOTE")

# # --- Baseline models ---
# models = {
#     "Logistic Regression": get_log_reg(class_weight="balanced"),
#     "Naive Bayes": get_naive_bayes()
# }

# results = {}

# # for name, model in models.items():
# #     print(f"\nTraining {name}...")
# #     model.fit(X_train_vec, y_train)

# #     # Predictions
# #     y_pred = model.predict(X_test_vec)


# #     # Metrics
# #     print(f"\n{name} Classification Report:")
# #     report = classification_report(y_test, y_pred, output_dict=True)
# #     print(classification_report(y_test, y_pred))

# #     # Confusion matrix
# #     cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
# #     plt.figure(figsize=(6,5))
# #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
# #     plt.title(f"{name} Confusion Matrix")
# #     plt.ylabel("True Label")
# #     plt.xlabel("Predicted Label")
# #     plt.tight_layout()
# #     # plt.savefig(f"results/{name.replace(' ', '_')}_confusion_matrix.png")
# #     plt.close()

# #     # Save results
# #     results[name] = {
# #         "macro_f1": report["macro avg"]["f1-score"]
# #     }

# # def measure_latency(model, X_test, n_runs=5):
# #     times = []
# #     for _ in range(n_runs):
# #         start = time.perf_counter()
# #         _ = model.predict(X_test)
# #         end = time.perf_counter()
# #         times.append((end - start) / X_test.shape[0])  # seconds per sample
# #     avg_time = np.mean(times) * 1000  # ms per sample
# #     return avg_time

# # # Example usage after training
# # logreg_latency = measure_latency(models["Logistic Regression"], X_test_vec)
# # nb_latency = measure_latency(models["Naive Bayes"], X_test_vec)

# # print(f"Logistic Regression latency: {logreg_latency:.4f} ms/message")
# # print(f"Naive Bayes latency: {nb_latency:.4f} ms/message")
# # # --- Print summary ---
# # print("\nBaseline Summary:")
# # for name, metrics in results.items():
# #     print(f"{name}: Macro-F1={metrics['macro_f1']:.3f}")
# # print(f"Logistic Regression latency: {logreg_latency:.4f} ms/message")
# # print(f"Naive Bayes latency: {nb_latency:.4f} ms/message")


# # ----BERT model----
# print("Training DistilBERT....")

# # Tokenizer

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# train_encodings = tokenizer(train_texts_small.tolist(), truncation=True, padding=True)
# test_encodings = tokenizer(test_texts_small.tolist(), truncation=True, padding=True)

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item

# train_dataset = Dataset(train_encodings, train_labels_small)
# test_dataset = Dataset(test_encodings, test_labels_small)
# # After train_test_split
# train_texts_small = [str(x) for x in train_texts_small]
# test_texts_small = [str(x) for x in test_texts_small]

# train_labels_small = train_labels_small.tolist()
# test_labels_small = test_labels_small.tolist()

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# training_args = TrainingArguments(
#     output_dir="./results",
#     #evaluation_strategy = "epoch",
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=100,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     report = classification_report(labels, preds, output_dict=True)
#     return {"macro_f1": report["macro avg"]["f1-score"]}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# # # Evaluate
# # bert_preds = trainer.predict(test_dataset)
# # bert_labels = test_dataset.labels
# # bert_report = classification_report(bert_labels, bert_preds.predictions.argmax(-1), output_dict=True)

# # print("DistilBERT Classification Report:")
# # print(classification_report(bert_labels, bert_preds.predictions.argmax(-1)))

# # results["DistilBERT"] = bert_report

# # # Save all results
# # df_results = pd.DataFrame(results).T
# # df_results.to_csv("baseline_results.csv")
# # print("All results saved to baseline_results.csv")

# # --- Evaluate ---
# metrics = trainer.evaluate()
# print("\n📊 DistilBERT Evaluation Metrics:")
# print(metrics)

# # --- Detailed Classification Report ---
# import numpy as np
# from sklearn.metrics import classification_report

# # Get predictions
# predictions = trainer.predict(test_dataset)
# y_pred = np.argmax(predictions.predictions, axis=1)
# y_true = predictions.label_ids

# print("\n📊 DistilBERT Classification Report:")
# print(classification_report(y_true, y_pred, digits=3))



#
#
#
#
#
#
#
#
#
#

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing import preprocess_text

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

from models import get_log_reg, get_naive_bayes

# --- Load unified dataset ---
def load_dataset(sample_size=None):
    df = pd.read_csv("toxic-spam-detection/data/unified_dataset.csv")
    
    # Optional: downsample for BERT training
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # apply preprocessing
    df["text"] = df["text"].astype(str).apply(preprocess_text)
    X = df['text']
    y = df['label']

    # CRITICAL FIX: Convert string labels to integers
    if y.dtype == 'object' or y.dtype == 'str':
        label_mapping = {'none': 0, 'toxic': 1, 'spam': 2, 'both': 3}
        y = y.map(label_mapping)
        print(f"✓ Converted string labels to integers using mapping: {label_mapping}")
    
    # Verify labels are integers
    assert y.dtype in ['int64', 'int32'], f"Labels must be integers, got {y.dtype}"
    
    # Stratified split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("=" * 50)
print("Loading datasets...")
print("=" * 50)

# Split into train/test
X_train, X_test, y_train, y_test = load_dataset()
train_texts_small, test_texts_small, train_labels_small, test_labels_small = load_dataset(sample_size=5000)

print(f"Full dataset: {len(X_train)} train, {len(X_test)} test")
print(f"Small dataset for BERT: {len(train_texts_small)} train, {len(test_texts_small)} test")
print(f"Label distribution: {y_train.value_counts().to_dict()}")

# --- TF-IDF representation ---
print("\n" + "=" * 50)
print("Creating TF-IDF vectors...")
print("=" * 50)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),  # unigrams + bigrams
    max_features=20000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"TF-IDF shape: {X_train_vec.shape}")

# --- Baseline models ---
models = {
    "Logistic Regression": get_log_reg(class_weight="balanced"),
    "Naive Bayes": get_naive_bayes()
}

results = {}

print("\n" + "=" * 50)
print("Training Baseline Models...")
print("=" * 50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_vec)
    
    # Metrics
    print(f"\n{name} Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    # Save results
    results[name] = {
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"]
    }

def measure_latency(model, X_test, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        end = time.perf_counter()
        times.append((end - start) / X_test.shape[0])  # seconds per sample
    avg_time = np.mean(times) * 1000  # ms per sample
    return avg_time

# Measure latency
logreg_latency = measure_latency(models["Logistic Regression"], X_test_vec)
nb_latency = measure_latency(models["Naive Bayes"], X_test_vec)

print("\n" + "=" * 50)
print("Baseline Summary:")
print("=" * 50)
for name, metrics in results.items():
    print(f"{name}: Macro-F1={metrics['macro_f1']:.3f}, Accuracy={metrics['accuracy']:.3f}")
print(f"Logistic Regression latency: {logreg_latency:.4f} ms/message")
print(f"Naive Bayes latency: {nb_latency:.4f} ms/message")

# ============================================================
# DISTILBERT MODEL
# ============================================================

print("\n" + "=" * 50)
print("Training DistilBERT...")
print("=" * 50)

# CRITICAL FIX: Convert pandas Series to list BEFORE tokenization
train_texts_list = train_texts_small.tolist()
test_texts_list = test_texts_small.tolist()
train_labels_list = train_labels_small.tolist()
test_labels_list = test_labels_small.tolist()

print(f"Preparing {len(train_texts_list)} training samples...")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize
print("Tokenizing texts...")
train_encodings = tokenizer(train_texts_list, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts_list, truncation=True, padding=True, max_length=128)

# CRITICAL FIX: Custom dataset class with different name
class ToxicityDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ToxicityDataset(train_encodings, train_labels_list)
test_dataset = ToxicityDataset(test_encodings, test_labels_list)

print(f"Created datasets: {len(train_dataset)} train, {len(test_dataset)} test")

# Load model
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=4
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", 
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,  # Added explicit learning rate
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

# --- Evaluate ---
print("\n" + "=" * 50)
print("Evaluating DistilBERT...")
print("=" * 50)

metrics = trainer.evaluate()
print("\n📊 DistilBERT Evaluation Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# --- Detailed Classification Report ---
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\n📊 DistilBERT Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# Confusion matrix for DistilBERT
cm_bert = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bert, annot=True, fmt="d", cmap="Greens")
plt.title("DistilBERT Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("results/DistilBERT_confusion_matrix.png")
plt.close()

# Save DistilBERT results
results["DistilBERT"] = {
    "macro_f1": metrics["eval_macro_f1"],
    "accuracy": metrics["eval_accuracy"]
}

# --- Save model ---
print("\nSaving model...")
trainer.save_model("./saved_models/distilbert")
tokenizer.save_pretrained("./saved_models/distilbert")

# --- Final comparison ---
print("\n" + "=" * 50)
print("FINAL RESULTS COMPARISON:")
print("=" * 50)
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values("macro_f1", ascending=False)
print(df_results)
df_results.to_csv("results/baseline_results.csv")

print("\n✅ All results saved!")
print("   - Confusion matrices: results/")
print("   - Summary table: results/baseline_results.csv")
print("   - Model checkpoint: saved_models/distilbert/")