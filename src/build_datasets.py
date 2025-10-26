import pandas as pd
from datasets import load_dataset



# load jigsaw data, sample of 500 rows due to storage issues
def load_jigsaw(path="toxic-spam-detection\data\jigsaw.csv"):
    df = pd.read_csv(path)[["comment_text", 'toxic']]
    df = df.rename(columns={'comment_text': 'text'})
    df['toxic_flag'] = df['toxic'].apply(lambda x: 1 if x==1 else 0)
    # df = df.sample(n=sample_size, random_state=42)
    df['spam_flag'] = 0
    return df

# civil comments: toxicity data
# def load_civil_comments():
#     ds = load_dataset("google/civil_comments")
#     df = pd.read_parquet(ds)
#     df['toxic_flag'] = (df['toxicity'] > 0.5).astype(int)
#     df = df[['text','toxic_flag']]
#     df['spam_flag'] = 0
#     return df

#
def load_youtube(path="toxic-spam-detection\data\youtoxic_english_1000.csv"):
    df = pd.read_csv(path)
    df = df[['Text', 'IsToxic']].rename(columns={"Text": "text", "IsToxic": "toxic_flag"})
    df['toxic_flag'] = df['toxic_flag'].apply(lambda x: 1 if x == 'TRUE' else 0)
    df['spam_flag'] = 0
    return df[['text','toxic_flag','spam_flag']]

def load_toxic_comments(path="toxic-spam-detection/data/toxicity_en.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={'is_toxic':'toxic_flag'})
    df['toxic_flag'] = df['toxic_flag'].apply(lambda x: 1 if x == 'Toxic' else 0)
    df['spam_flag'] = 0
    return df[['text','toxic_flag','spam_flag']]

def load_sms_spam(path="toxic-spam-detection\data\spam.csv"):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'spam_flag', 'v2': 'text'})
    df['toxic_flag'] = 0
    df['spam_flag'] = df['spam_flag'].apply(lambda x: 1 if x == 'spam' else 0)
    return df[['text','toxic_flag','spam_flag']]

def load_email_spam(path="toxic-spam-detection\data\emails.csv"):
    df = pd.read_csv(path)
    df = df[["text", "spam"]].rename(columns={"spam": "spam_flag"})
    df['toxic_flag'] = 0
    return df[['text','toxic_flag','spam_flag']]

def map_labels(df):
    def assign(row):
        if row['toxic_flag']==1:
            return "toxic"
        elif row['spam_flag']==1:
            return "spam"
        else:
            return "none"
    df['label'] = df.apply(assign, axis=1)
    return df[['text','label']]

if __name__ == "__main__":
    jigsaw = load_jigsaw()
    # civil = load_civil_comments()
    youtube = load_youtube()
    toxic_comments = load_toxic_comments()
    sms = load_sms_spam()
    email = load_email_spam()

    datasets = [jigsaw, youtube, toxic_comments, sms, email]
    df = pd.concat(datasets, ignore_index=True)
    df = map_labels(df)

    print("✅ Final label distribution:")
    print(df['label'].value_counts())

    df.to_csv("toxic-spam-detection/data/unified_dataset.csv", index=False)
    print("💾 Saved unified dataset to data/unified_dataset.csv")






