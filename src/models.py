
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoModelForSequenceClassification

def get_log_reg(class_weight=None):
    return LogisticRegression(max_iter=1000, class_weight=class_weight)

def get_naive_bayes():
    return MultinomialNB()

def get_bert_model(model_name="distilbert-base-uncased", num_labels=4):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
