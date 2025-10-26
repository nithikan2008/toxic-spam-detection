import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
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

print("=" * 70)
print("CROSS-PLATFORM ROBUSTNESS TESTING")
print("=" * 70)
print("Testing generalization across communication platforms:")
print("  - Discord (gaming culture, voice chat slang)")
print("  - WhatsApp (more formal, international users)")
print("  - TikTok/Instagram (Gen Z slang, extreme abbreviations)")
print()

# ==========================================
# CREATE PLATFORM-SPECIFIC TEST SETS
# ==========================================

# DISCORD - Gaming culture, voice chat lingo, aggressive but not always toxic
discord_samples = [
    # Toxic gaming rage
    ("gg ez you're trash", "toxic"),
    ("stop camping loser", "toxic"),
    ("kys you're so bad", "toxic"),
    ("imagine losing to bots smh", "toxic"),
    ("touch grass loser", "toxic"),
    ("hardstuck silver lmao", "toxic"),
    ("nah ur strats worthless", "toxic"),
    ("ratio + L + you fell off", "toxic"),
    ("lol delete this game clown", "toxic"),
    
    # Gaming spam (often crypto/boosting services)
    ("Join my new Minecraft server! Link:", "spam"),
    ("Free robux click here ->", "spam"),
    ("giveaway night! dm me for keys", "spam"),
    ("selling valorant accounts cheap", "spam"),
    ("Free vbucks generator working!", "spam"),
    ("vote for us on top.gg", "spam"),
    ("join clan alpha now", "spam"),
    # Both (toxic + spam combo)
    ("ur trash join my clan to learn", "both"),
    ("noobs everyone click for real gamers", "both"),
    ("literal bot check my vids", "both"),
    ("lol ur build is junk buy this one", "both"),
    ("join our pro discord not these clowns", "both"),
    
    # Clean gaming communication (can seem aggressive but isn't)
    ("gg wp everyone", "none"),
    ("anyone up for ranked later?", "none"),
    ("brb grabbin snacks", "none"),
    ("clutch or kick", "none"),
    ("lets run it back", "none"),
    ("im lagging so bad rn", "none"),
    ("what server you on", "none"),
    ("add me for duos", "none"),
    ("that was a crazy play", "none"),
    ("ult is ready", "none"),
]

# WHATSAPP - More formal, international, family/work groups
whatsapp_samples = [
    # Toxic (more subtle, passive-aggressive)
    ("You're so annoying, fuck you.", "toxic"),
    ("Why are you so dumb everytime?", "toxic"),
    ("Your opinion is worthless.", "toxic"),
    ("Ugh, you ruin everything.", "toxic"),
    ("You clearly don't understand anything.", "toxic"),
    ("Stop embarrassing yourself.", "toxic"),
    
    # Spam (common WhatsApp spam types)
    ("Congratulations you won 10000 dollars! Claim now", "spam"),
    ("Hey, check this link out!", "spam"),
    ("Your package is waiting! Click link to confirm", "spam"),
    ("Limited time offer 80% off, don't miss out!", "spam"),
    ("Your account will be blocked, verify account here:", "spam"),
    ("Forward this message to 10 people for good luck: ", "spam"),
    ("No experience job opportunity, click link to apply!", "spam"),
    ("Bitcoin investment guaranteed returns, please message me for details.", "spam"),

    # Both (toxic + spam combo)
    ("You're useless, buy this to fix that:", "both"),
    ("Stop texting me, use this app", "both"),
    ("Quit whining, check this now!", "both"),
    ("Annoying much? Get a life with this:", "both"),
    
    # Clean (formal communication)
    ("Hey, how are you?", "none"),
    ("Meeting is scheduled for 3pm tomorrow.", "none"),
    ("Let's catch up soon, I haven't seen you in a while!", "none"),
    ("Happy birthday! Hope you have a great day!", "none"),
    ("Thanks for your help earlier.", "none"),
    ("Let me know if you need anything else!", "none"),
    ("See you at the meeting tomorrow.", "none"),
    ("Could you send me your address please?", "none"),
    ("Sorry for the late response!", "none"),
    ("Have a safe trip.", "none"),
]

# TIKTOK/INSTAGRAM - Gen Z slang, heavy abbreviations, memetic
instagram_samples = [
    # Toxic (Gen Z style insults)
    ("ngl go fuck yourself you piece of shit", "toxic"),
    ("bro u serious w this shi? smh", "toxic"),
    ("lmfao what a clown", "toxic"),
    ("fuck you jackass", "toxic"),
    ("bitch who even asked", "toxic"),
    ("the delusion is real", "toxic"),
    ("fucking retard", "toxic"),
    
    # Spam (typical social media spam)
    ("Dm me to collab!", "spam"),
    ("Get 10k followers now:", "spam"),
    ("Exclusive merch drop don't miss out!", "spam"),
    ("Love your vibe! Try our product here:", "spam"),
    ("Click link in bio to win iphone", "spam"),
    ("Earn money posting pics right now", "spam"),
    ("Get followers fast!", "spam"),
    
    # Both (toxic + spam combo)
    ("You look like a bot buy followers instead", "both"),
    ("Your content is trash check mine instead", "both"),
    ("Your makeup sucks, buy this:", "both"),

    # Clean (casual but friendly)
    ("same bruh so real", "none"),
    ("no bc literally same", "none"),
    ("obsessed with this", "none"),
    ("giving main character energy fr", "none"),
    ("the way i screamed", "none"),
    ("not me crying over this", "none"),
    ("ngl so valid", "none"),
    ("i fw this frfr", "none"),
    ("aesthetic is everything", "none"),
    ("ilysm bestie", "none"),
    ("youre so pretty omg baddie", "none"),
    ("need this in my life", "none"),
]

# Create DataFrames
df_discord = pd.DataFrame(discord_samples, columns=['text', 'label'])
df_discord['platform'] = 'Discord'
df_discord['label'] = df_discord['label'].map(LABEL_MAP)

df_whatsapp = pd.DataFrame(whatsapp_samples, columns=['text', 'label'])
df_whatsapp['platform'] = 'WhatsApp'
df_whatsapp['label'] = df_whatsapp['label'].map(LABEL_MAP)

df_insta = pd.DataFrame(instagram_samples, columns=['text', 'label'])
df_insta['platform'] = 'Instagram'
df_insta['label'] = df_insta['label'].map(LABEL_MAP)

print("Platform Test Sets Created:")
print(f"  Discord: {len(df_discord)} samples")
print(f"    Distribution: {df_discord['label'].map(REVERSE_MAP).value_counts().to_dict()}")
print(f"  WhatsApp: {len(df_whatsapp)} samples")
print(f"    Distribution: {df_whatsapp['label'].map(REVERSE_MAP).value_counts().to_dict()}")
print(f"  TikTok/Instagram: {len(df_insta)} samples")
print(f"    Distribution: {df_insta['label'].map(REVERSE_MAP).value_counts().to_dict()}")
print()

# Save test sets
df_discord.to_csv("data/discord_test.csv", index=False)
df_whatsapp.to_csv("data/whatsapp_test.csv", index=False)
df_insta.to_csv("data/tiktok_test.csv", index=False)

# ==========================================
# LOAD MODELS (trained on generic dataset)
# ==========================================

print("=" * 70)
print("LOADING MODELS (trained on Reddit/SMS/Jigsaw data)")
print("=" * 70)

# Load training data
df_train = pd.read_csv("data/unified_dataset.csv")
df_train['text'] = df_train['text'].astype(str).apply(preprocess_text)

if df_train['label'].dtype == 'object':
    df_train['label'] = df_train['label'].map(LABEL_MAP)

# Train Logistic Regression
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
bert_model = DistilBertForSequenceClassification.from_pretrained("./saved_models/distilbert")
bert_tokenizer = DistilBertTokenizer.from_pretrained("./saved_models/distilbert")
bert_model.eval()
print("✓ DistilBERT loaded\n")

# ==========================================
# TEST ON EACH PLATFORM
# ==========================================

def test_platform(df_platform, platform_name):
    """Test both models on a platform-specific dataset"""
    
    print("=" * 70)
    print(f"TESTING ON {platform_name.upper()}")
    print("=" * 70)
    
    # Preprocess
    df_platform['text_processed'] = df_platform['text'].apply(preprocess_text)
    y_true = df_platform['label']
    
    # Test Logistic Regression
    X_vec = vectorizer.transform(df_platform['text_processed'])
    y_pred_lr = lr_model.predict(X_vec)
    
    lr_acc = accuracy_score(y_true, y_pred_lr)
    lr_f1 = f1_score(y_true, y_pred_lr, average='macro')
    
    print(f"\nLogistic Regression on {platform_name}:")
    print(f"  Accuracy: {lr_acc:.3f}")
    print(f"  Macro-F1: {lr_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_lr, 
                                target_names=['none', 'toxic', 'spam', 'both'],
                                labels=[0, 1, 2, 3],
                                zero_division=0))
    
    # Test DistilBERT
    inputs = bert_tokenizer(
        df_platform['text_processed'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        y_pred_bert = torch.argmax(outputs.logits, dim=1).numpy()
    
    bert_acc = accuracy_score(y_true, y_pred_bert)
    bert_f1 = f1_score(y_true, y_pred_bert, average='macro')
    
    print(f"\nDistilBERT on {platform_name}:")
    print(f"  Accuracy: {bert_acc:.3f}")
    print(f"  Macro-F1: {bert_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bert,
                                target_names=['none', 'toxic', 'spam', 'both'],
                                labels=[0, 1, 2, 3],
                                zero_division=0))
    
    return {
        'platform': platform_name,
        'lr_acc': lr_acc,
        'lr_f1': lr_f1,
        'bert_acc': bert_acc,
        'bert_f1': bert_f1,
        'y_true': y_true,
        'y_pred_lr': y_pred_lr,
        'y_pred_bert': y_pred_bert
    }

# Test each platform
results_discord = test_platform(df_discord, "Discord")
results_whatsapp = test_platform(df_whatsapp, "WhatsApp")
results_insta = test_platform(df_insta, "Instagram")

# ==========================================
# COMPARISON & VISUALIZATION
# ==========================================

print("\n" + "=" * 70)
print("CROSS-PLATFORM PERFORMANCE SUMMARY")
print("=" * 70)

# Original performance (trained on Reddit/SMS/Jigsaw)
original_lr_f1 = 0.815
original_bert_f1 = 0.712

results_df = pd.DataFrame([
    {
        'Platform': 'Original\n(Reddit/SMS)',
        'LR_F1': original_lr_f1,
        'BERT_F1': original_bert_f1,
        'LR_Drop': 0,
        'BERT_Drop': 0
    },
    {
        'Platform': 'Discord',
        'LR_F1': results_discord['lr_f1'],
        'BERT_F1': results_discord['bert_f1'],
        'LR_Drop': ((original_lr_f1 - results_discord['lr_f1']) / original_lr_f1) * 100,
        'BERT_Drop': ((original_bert_f1 - results_discord['bert_f1']) / original_bert_f1) * 100
    },
    {
        'Platform': 'WhatsApp',
        'LR_F1': results_whatsapp['lr_f1'],
        'BERT_F1': results_whatsapp['bert_f1'],
        'LR_Drop': ((original_lr_f1 - results_whatsapp['lr_f1']) / original_lr_f1) * 100,
        'BERT_Drop': ((original_bert_f1 - results_whatsapp['bert_f1']) / original_bert_f1) * 100
    },
    {
        'Platform': 'Instagram',
        'LR_F1': results_insta['lr_f1'],
        'BERT_F1': results_insta['bert_f1'],
        'LR_Drop': ((original_lr_f1 - results_insta['lr_f1']) / original_lr_f1) * 100,
        'BERT_Drop': ((original_bert_f1 - results_insta['bert_f1']) / original_bert_f1) * 100
    }
])

print("\n" + results_df.to_string(index=False))
print()

# Find which model degrades less
avg_lr_drop = results_df[results_df['Platform'] != 'Original\n(Reddit/SMS)']['LR_Drop'].mean()
avg_bert_drop = results_df[results_df['Platform'] != 'Original\n(Reddit/SMS)']['BERT_Drop'].mean()

print(f"Average Performance Drop Across Platforms:")
print(f"  Logistic Regression: {avg_lr_drop:.1f}%")
print(f"  DistilBERT: {avg_bert_drop:.1f}%")
print()

if avg_lr_drop < avg_bert_drop:
    print(f"🔑 KEY FINDING: Logistic Regression shows better cross-platform generalization")
    print(f"   ({avg_lr_drop:.1f}% avg drop vs {avg_bert_drop:.1f}% for DistilBERT)")
else:
    print(f"🔑 KEY FINDING: DistilBERT shows better cross-platform generalization")
    print(f"   ({avg_bert_drop:.1f}% avg drop vs {avg_lr_drop:.1f}% for Logistic Regression)")

# ==========================================
# VISUALIZATIONS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: F1 Score by Platform
ax1 = axes[0, 0]
x = np.arange(len(results_df))
width = 0.35
ax1.bar(x - width/2, results_df['LR_F1'], width, label='Logistic Regression', alpha=0.8, color='#3498db')
ax1.bar(x + width/2, results_df['BERT_F1'], width, label='DistilBERT', alpha=0.8, color='#e74c3c')
ax1.set_ylabel('Macro-F1 Score', fontsize=11)
ax1.set_title('Cross-Platform Performance Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Platform'], fontsize=9)
ax1.legend()
ax1.set_ylim([0, 1])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Plot 2: Performance Drop
ax2 = axes[0, 1]
platforms = results_df[results_df['Platform'] != 'Original\n(Reddit/SMS)']['Platform']
lr_drops = results_df[results_df['Platform'] != 'Original\n(Reddit/SMS)']['LR_Drop']
bert_drops = results_df[results_df['Platform'] != 'Original\n(Reddit/SMS)']['BERT_Drop']
x2 = np.arange(len(platforms))
ax2.bar(x2 - width/2, lr_drops, width, label='Logistic Regression', alpha=0.8, color='#3498db')
ax2.bar(x2 + width/2, bert_drops, width, label='DistilBERT', alpha=0.8, color='#e74c3c')
ax2.set_ylabel('Performance Drop (%)', fontsize=11)
ax2.set_title('Cross-Platform Degradation', fontsize=13, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(platforms, fontsize=9)
ax2.legend()

# Plot 3: Confusion Matrix - Discord
ax3 = axes[1, 0]
cm_discord = confusion_matrix(results_discord['y_true'], results_discord['y_pred_lr'], labels=[0, 1, 2, 3])
sns.heatmap(cm_discord, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['none', 'toxic', 'spam', 'both'],
            yticklabels=['none', 'toxic', 'spam', 'both'])
ax3.set_title('LR on Discord - Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# Plot 4: Accuracy comparison
ax4 = axes[1, 1]
metrics = ['Accuracy', 'Macro-F1']
lr_metrics = [
    np.mean([results_discord['lr_acc'], results_whatsapp['lr_acc'], results_insta['lr_acc']]),
    np.mean([results_discord['lr_f1'], results_whatsapp['lr_f1'], results_insta['lr_f1']])
]
bert_metrics = [
    np.mean([results_discord['bert_acc'], results_whatsapp['bert_acc'], results_insta['bert_acc']]),
    np.mean([results_discord['bert_f1'], results_whatsapp['bert_f1'], results_insta['bert_f1']])
]
x3 = np.arange(len(metrics))
ax4.bar(x3 - width/2, lr_metrics, width, label='Logistic Regression', alpha=0.8, color='#3498db')
ax4.bar(x3 + width/2, bert_metrics, width, label='DistilBERT', alpha=0.8, color='#e74c3c')
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Average Performance Across All Platforms', fontsize=12, fontweight='bold')
ax4.set_xticks(x3)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('results/cross_platform_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to results/cross_platform_analysis.png")

# Save results
results_df.to_csv('results/cross_platform_results.csv', index=False)
print("✓ Saved results to results/cross_platform_results.csv")

print("\n" + "=" * 70)
print("CROSS-PLATFORM TESTING COMPLETE! ✅")
print("=" * 70)