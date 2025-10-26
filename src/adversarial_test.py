import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from preprocessing import preprocess_text
import matplotlib.pyplot as plt
import seaborn as sns

# Label mapping
LABEL_MAP = {'none': 0, 'toxic': 1, 'spam': 2, 'both': 3}
REVERSE_MAP = {0: 'none', 1: 'toxic', 2: 'spam', 3: 'both'}

# ==========================================
# CREATE ADVERSARIAL TEST SET (50 samples)
# ==========================================

adversarial_samples = [
    # OBFUSCATION - Leetspeak (15 samples)
    ("fr33 m0n3y cl1ck here n0w", "spam"),
    ("y0u suck 1d10t", "toxic"),
    ("g3t rich qu1ck", "spam"),
    ("you're a d-u-m-b-a-s-s", "toxic"),
    ("w1n fr33 1ph0ne", "spam"),
    ("k1ll y0urself", "toxic"),
    ("ea5y ca5h", "spam"),
    ("you f-u-c-k-i-n-g idiot", "toxic"),
    ("earn m0ney f4st", "spam"),
    ("h0e go d1e", "toxic"),
    ("ca$h prize w1n", "spam"),
    ("you're tr@sh", "toxic"),
    ("fr33 g1ft cards", "spam"),
    ("ret@rded piece of garbage", "toxic"),
    ("make m0ney fr0m h0me", "spam"),
    
    # EXTREME SPACING/PUNCTUATION (15 samples)
    ("f.r.e.e m.o.n.e.y", "spam"),
    ("y o u  s u c k", "toxic"),
    ("c-l-i-c-k h-e-r-e", "spam"),
    ("i d i o t", "toxic"),
    ("w i n. n o w", "spam"),
    ("k i l l yourself", "toxic"),
    ("c a s h prize", "spam"),
    ("you are t r a s h", "toxic"),
    ("free!!! money!!! now!!!", "spam"),
    ("die die die die", "toxic"),
    ("$$$CASH$$$", "spam"),
    ("stupid stupid stupid", "toxic"),
    ("FREE MONEY FREE MONEY", "spam"),
    ("hate hate hate you", "toxic"),
    ("!!!CLICK HERE!!!", "spam"),
    
    # EMOJI/SYMBOL SUBSTITUTION (10 samples)
    ("💰💰💰 click here 👇", "spam"),
    ("🤬🤬🤬", "toxic"),
    ("💵 win now 🎁", "spam"),
    ("you're 💩", "toxic"),
    ("🔥 hot deal 🔥", "spam"),
    ("go 💀 yourself", "toxic"),
    ("💸 easy money 💸", "spam"),
    ("🖕🖕🖕", "toxic"),
    ("🎉 free gift 🎉", "spam"),
    ("i hope you 💀", "toxic"),
    
    # CLEAN SAMPLES (8 samples - should maintain accuracy)
    ("Great product, highly recommend!", "none"),
    ("Thanks for your help today", "none"),
    ("The weather is nice", "none"),
    ("I enjoyed the movie", "none"),
    ("Have a great day", "none"),
    ("This is informative", "none"),
    ("Good morning everyone", "none"),
    ("Interesting perspective", "none"),
    
    # BOTH TOXIC + SPAM (2 samples - for completeness)
    ("you idiot click here for free money", "both"),
    ("stupid loser win cash now", "both"),
]

# Convert to DataFrame
df_adv = pd.DataFrame(adversarial_samples, columns=['text', 'label'])
df_adv['label'] = df_adv['label'].map(LABEL_MAP)

print("=" * 60)
print("ADVERSARIAL TEST SET CREATED")
print("=" * 60)
print(f"Total samples: {len(df_adv)}")
print(f"Label distribution:\n{df_adv['label'].map(REVERSE_MAP).value_counts()}")
print()

# Save adversarial dataset
df_adv.to_csv("data/adversarial_test.csv", index=False)
print("✓ Saved to data/adversarial_test.csv\n")

# ==========================================
# LOAD MODELS
# ==========================================

print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

# Load training data for vectorizer
df_train = pd.read_csv("data/unified_dataset.csv")
df_train['text'] = df_train['text'].astype(str).apply(preprocess_text)

# Map labels if they're strings
if df_train['label'].dtype == 'object':
    df_train['label'] = df_train['label'].map(LABEL_MAP)

# Train Logistic Regression (if not saved)
print("Training Logistic Regression...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)
X_train_vec = vectorizer.fit_transform(df_train['text'])
y_train = df_train['label']

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_vec, y_train)
print("✓ Logistic Regression trained\n")

# Load DistilBERT
print("Loading DistilBERT...")
try:
    bert_model = DistilBertForSequenceClassification.from_pretrained("./saved_models/distilbert")
    bert_tokenizer = DistilBertTokenizer.from_pretrained("./saved_models/distilbert")
    print("✓ DistilBERT loaded from saved_models/\n")
except:
    print("⚠ Could not load from saved_models/, loading base model")
    bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
    bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("⚠ Using base DistilBERT (not fine-tuned)\n")

bert_model.eval()

# ==========================================
# TEST LOGISTIC REGRESSION
# ==========================================

print("=" * 60)
print("TESTING LOGISTIC REGRESSION ON ADVERSARIAL DATA")
print("=" * 60)

# Preprocess adversarial texts
df_adv['text_processed'] = df_adv['text'].apply(preprocess_text)
X_adv_vec = vectorizer.transform(df_adv['text_processed'])
y_adv = df_adv['label']

# Predict
y_pred_lr = lr_model.predict(X_adv_vec)

# Metrics
lr_acc = accuracy_score(y_adv, y_pred_lr)
lr_macro_f1 = f1_score(y_adv, y_pred_lr, average='macro')

print(f"Accuracy: {lr_acc:.3f}")
print(f"Macro-F1: {lr_macro_f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_adv, y_pred_lr, target_names=['none', 'toxic', 'spam', 'both'], labels=[0, 1, 2, 3]))

# ==========================================
# TEST DISTILBERT
# ==========================================

print("=" * 60)
print("TESTING DISTILBERT ON ADVERSARIAL DATA")
print("=" * 60)

# Tokenize
inputs = bert_tokenizer(
    df_adv['text_processed'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Predict
with torch.no_grad():
    outputs = bert_model(**inputs)
    y_pred_bert = torch.argmax(outputs.logits, dim=1).numpy()

# Metrics
bert_acc = accuracy_score(y_adv, y_pred_bert)
bert_macro_f1 = f1_score(y_adv, y_pred_bert, average='macro')

print(f"Accuracy: {bert_acc:.3f}")
print(f"Macro-F1: {bert_macro_f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_adv, y_pred_bert, target_names=['none', 'toxic', 'spam', 'both'], labels=[0, 1, 2, 3]))

# ==========================================
# COMPARISON & VISUALIZATION
# ==========================================

print("\n" + "=" * 60)
print("ROBUSTNESS COMPARISON")
print("=" * 60)

# Original performance (from your training)
lr_original_f1 = 0.815
bert_original_f1 = 0.712

# Calculate performance drop
lr_drop = ((lr_original_f1 - lr_macro_f1) / lr_original_f1) * 100
bert_drop = ((bert_original_f1 - bert_macro_f1) / bert_original_f1) * 100

print(f"\nLogistic Regression:")
print(f"  Original Macro-F1: {lr_original_f1:.3f}")
print(f"  Adversarial Macro-F1: {lr_macro_f1:.3f}")
print(f"  Performance Drop: {lr_drop:.1f}%")

print(f"\nDistilBERT:")
print(f"  Original Macro-F1: {bert_original_f1:.3f}")
print(f"  Adversarial Macro-F1: {bert_macro_f1:.3f}")
print(f"  Performance Drop: {bert_drop:.1f}%")

print(f"\n🔑 KEY FINDING:")
if bert_drop < lr_drop:
    print(f"   DistilBERT is MORE ROBUST ({bert_drop:.1f}% drop vs {lr_drop:.1f}% drop)")
    print(f"   Difference: {lr_drop - bert_drop:.1f} percentage points")
else:
    print(f"   Logistic Regression is more robust ({lr_drop:.1f}% drop vs {bert_drop:.1f}% drop)")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Macro-F1 comparison
models = ['LR\n(Clean)', 'LR\n(Adversarial)', 'BERT\n(Clean)', 'BERT\n(Adversarial)']
f1_scores = [lr_original_f1, lr_macro_f1, bert_original_f1, bert_macro_f1]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#e67e22']

axes[0].bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Macro-F1 Score', fontsize=12)
axes[0].set_title('Model Performance: Clean vs Adversarial Data', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(f1_scores):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 2: Performance drop
drop_models = ['Logistic\nRegression', 'DistilBERT']
drop_values = [lr_drop, bert_drop]
colors_drop = ['#e74c3c' if lr_drop > bert_drop else '#95a5a6', 
               '#e74c3c' if bert_drop > lr_drop else '#95a5a6']

axes[1].bar(drop_models, drop_values, color=colors_drop, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Performance Drop (%)', fontsize=12)
axes[1].set_title('Robustness to Adversarial Obfuscation', fontsize=14, fontweight='bold')
for i, v in enumerate(drop_values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/adversarial_robustness_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved plot to results/adversarial_robustness_comparison.png")

# Save results summary
results_summary = {
    'Model': ['Logistic Regression', 'DistilBERT'],
    'Clean_MacroF1': [lr_original_f1, bert_original_f1],
    'Adversarial_MacroF1': [lr_macro_f1, bert_macro_f1],
    'Performance_Drop_%': [lr_drop, bert_drop],
    'Adversarial_Accuracy': [lr_acc, bert_acc]
}

df_results = pd.DataFrame(results_summary)
df_results.to_csv('results/adversarial_results.csv', index=False)
print("✓ Saved results to results/adversarial_results.csv")

print("\n" + "=" * 60)
print("ADVERSARIAL TESTING COMPLETE! ✅")
print("=" * 60)