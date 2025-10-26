import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from preprocessing import preprocess_text
import time

# Page config
st.set_page_config(
    page_title="Content Moderation AI",
    page_icon="🛡️",
    layout="wide"
)

# Label mapping
LABEL_MAP = {'none': 0, 'toxic': 1, 'spam': 2, 'both': 3}
REVERSE_MAP = {0: 'none', 1: 'toxic', 2: 'spam', 3: 'both'}

# Styling for labels
LABEL_COLORS = {
    'none': '#2ecc71',      # Green
    'toxic': '#e74c3c',     # Red
    'spam': '#f39c12',      # Orange
    'both': '#8e44ad'       # Purple
}

LABEL_EMOJIS = {
    'none': '✅',
    'toxic': '⚠️',
    'spam': '📧',
    'both': '🚨'
}

LABEL_DESCRIPTIONS = {
    'none': 'Clean Content - Safe to display',
    'toxic': 'Toxic Content - Contains harmful or offensive language',
    'spam': 'Spam Content - Promotional or unsolicited message',
    'both': 'Toxic + Spam - Contains both offensive language and spam'
}

# Cache models
@st.cache_resource
def load_models():
    """Load both LR and DistilBERT models"""
    
    # Load training data for LR
    df_train = pd.read_csv("data/unified_dataset.csv")
    df_train['text'] = df_train['text'].astype(str).apply(preprocess_text)
    
    if df_train['label'].dtype == 'object':
        df_train['label'] = df_train['label'].map(LABEL_MAP)
    
    # Train Logistic Regression
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    X_train = vectorizer.fit_transform(df_train['text'])
    y_train = df_train['label']
    
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Load DistilBERT
    bert_model = DistilBertForSequenceClassification.from_pretrained("./saved_models/distilbert")
    bert_tokenizer = DistilBertTokenizer.from_pretrained("./saved_models/distilbert")
    bert_model.eval()
    
    return lr_model, vectorizer, bert_model, bert_tokenizer

# Load models
with st.spinner("Loading AI models..."):
    lr_model, vectorizer, bert_model, bert_tokenizer = load_models()

# Header
st.title("🛡️ Social Media Content Moderation")
st.markdown("**AI-powered toxicity and spam detection** | Independent Research Project")
st.markdown("---")

# Sidebar - Info
with st.sidebar:
    st.header("About This Demo")
    st.markdown("""
    This tool demonstrates two approaches to content moderation:
    
    **Logistic Regression**
    - Fast (0.1ms per message)
    - Robust to obfuscation
    - Interpretable features
    
    **DistilBERT (Transformer)**
    - Context-aware
    - Better on clean data
    - Slower but more accurate
    
    **Research Findings:**
    - LR is 2x more robust to adversarial attacks
    - BERT slightly better at cross-platform generalization
    - LR recommended for production (speed + robustness)
    """)
    
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("LR Macro-F1", "0.815")
    st.metric("BERT Macro-F1", "0.712")
    
    st.markdown("---")
    st.info("Try adversarial examples like: 'fr33 m0n3y' or 'y0u suck'")

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Model Comparison", "Example Tests"])

# TAB 1: Single Prediction
with tab1:
    st.header("Test a Message")
    
    # Model selection
    model_choice = st.radio(
        "Choose Model:",
        ["Logistic Regression (Fast)", "DistilBERT (Accurate)"],
        horizontal=True
    )
    
    # Text input
    user_input = st.text_area(
        "Enter message to analyze:",
        height=120,
        placeholder="Type or paste a message here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if analyze_button and user_input:
        # Preprocess
        processed_text = preprocess_text(user_input)
        
        # Predict based on model choice
        start_time = time.time()
        
        if "Logistic" in model_choice:
            # Logistic Regression
            X_vec = vectorizer.transform([processed_text])
            prediction = lr_model.predict(X_vec)[0]
            probs = lr_model.predict_proba(X_vec)[0]
            
            # Handle case where model doesn't have all 4 classes
            # Pad probabilities array to length 4
            probabilities = np.zeros(4)
            for i, class_idx in enumerate(lr_model.classes_):
                probabilities[class_idx] = probs[i]
            
            # Get top features
            feature_names = vectorizer.get_feature_names_out()
            text_vec = X_vec.toarray()[0]
            top_features_idx = text_vec.argsort()[-5:][::-1]
            top_features = [(feature_names[i], text_vec[i]) for i in top_features_idx if text_vec[i] > 0]
            
        else:
            # DistilBERT
            inputs = bert_tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0].numpy()
                prediction = probabilities.argmax()
            
            # Get attention to tokens (simplified)
            tokens = bert_tokenizer.tokenize(processed_text)
            top_features = [(token, 1.0) for token in tokens[:5]]  # Simplified for demo
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Display results
        st.markdown("---")
        
        predicted_label = REVERSE_MAP[prediction]
        label_color = LABEL_COLORS[predicted_label]
        label_emoji = LABEL_EMOJIS[predicted_label]
        
        # Big result card
        st.markdown(f"""
        <div style="
            padding: 30px;
            border-radius: 10px;
            border: 3px solid {label_color};
            background: linear-gradient(135deg, {label_color}22 0%, {label_color}11 100%);
            text-align: center;
        ">
            <h1 style="color: {label_color}; margin: 0;">
                {label_emoji} {predicted_label.upper()}
            </h1>
            <p style="font-size: 18px; color: #666; margin-top: 10px;">
                {LABEL_DESCRIPTIONS[predicted_label]}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Confidence scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confidence Scores")
            for i, label in enumerate(['none', 'toxic', 'spam', 'both']):
                confidence = probabilities[i] * 100
                color = LABEL_COLORS[label]
                
                # Progress bar with custom styling
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: bold;">{LABEL_EMOJIS[label]} {label.capitalize()}</span>
                        <span style="color: {color}; font-weight: bold;">{confidence:.1f}%</span>
                    </div>
                    <div style="
                        background: #e0e0e0;
                        border-radius: 10px;
                        height: 25px;
                        overflow: hidden;
                    ">
                        <div style="
                            background: {color};
                            width: {confidence}%;
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Key Features Detected")
            
            if top_features:
                st.markdown("**Most influential words/patterns:**")
                for i, (feature, weight) in enumerate(top_features[:5], 1):
                    st.markdown(f"`{i}.` **{feature}**")
            else:
                st.info("No strong features detected")
            
            st.markdown("")
            st.metric("Inference Time", f"{inference_time:.2f} ms")
    
    elif analyze_button:
        st.warning("⚠️ Please enter a message to analyze")

# TAB 2: Model Comparison
with tab2:
    st.header("Compare Both Models")
    st.markdown("See how Logistic Regression and DistilBERT perform on the same input")
    
    comparison_input = st.text_area(
        "Enter message for comparison:",
        height=100,
        key="comparison_input",
        placeholder="Try: 'fr33 m0n3y' or 'you're an idiot'"
    )
    
    if st.button("Compare Models", type="primary"):
        if comparison_input:
            processed = preprocess_text(comparison_input)
            
            col1, col2 = st.columns(2)
            
            # Logistic Regression
            with col1:
                st.subheader("Logistic Regression")
                
                start = time.time()
                X_vec = vectorizer.transform([processed])
                lr_pred = lr_model.predict(X_vec)[0]
                lr_probs_raw = lr_model.predict_proba(X_vec)[0]
                lr_time = (time.time() - start) * 1000
                
                # Pad probabilities to length 4
                lr_probs = np.zeros(4)
                for i, class_idx in enumerate(lr_model.classes_):
                    lr_probs[class_idx] = lr_probs_raw[i]
                
                lr_label = REVERSE_MAP[lr_pred]
                lr_color = LABEL_COLORS[lr_label]
                
                st.markdown(f"""
                <div style="
                    padding: 20px;
                    border-radius: 8px;
                    border: 2px solid {lr_color};
                    background: {lr_color}15;
                    text-align: center;
                ">
                    <h2 style="color: {lr_color}; margin: 0;">
                        {LABEL_EMOJIS[lr_label]} {lr_label.upper()}
                    </h2>
                    <p style="margin-top: 10px;">Confidence: {lr_probs[lr_pred]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Inference Time", f"{lr_time:.3f} ms")
                
                with st.expander("View all probabilities"):
                    for i, label in enumerate(['none', 'toxic', 'spam', 'both']):
                        st.write(f"{label}: {lr_probs[i]*100:.1f}%")
            
            # DistilBERT
            with col2:
                st.subheader("DistilBERT")
                
                start = time.time()
                inputs = bert_tokenizer(
                    processed,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                )
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    bert_probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
                    bert_pred = bert_probs.argmax()
                bert_time = (time.time() - start) * 1000
                
                bert_label = REVERSE_MAP[bert_pred]
                bert_color = LABEL_COLORS[bert_label]
                
                st.markdown(f"""
                <div style="
                    padding: 20px;
                    border-radius: 8px;
                    border: 2px solid {bert_color};
                    background: {bert_color}15;
                    text-align: center;
                ">
                    <h2 style="color: {bert_color}; margin: 0;">
                        {LABEL_EMOJIS[bert_label]} {bert_label.upper()}
                    </h2>
                    <p style="margin-top: 10px;">Confidence: {bert_probs[bert_pred]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Inference Time", f"{bert_time:.2f} ms")
                
                with st.expander("View all probabilities"):
                    for i, label in enumerate(['none', 'toxic', 'spam', 'both']):
                        st.write(f"{label}: {bert_probs[i]*100:.1f}%")
            
            # Speed comparison
            st.markdown("---")
            st.subheader("Speed Comparison")
            speed_ratio = bert_time / lr_time
            st.info(f"**DistilBERT is {speed_ratio:.1f}x slower** than Logistic Regression ({bert_time:.2f}ms vs {lr_time:.3f}ms)")
            
            # Agreement
            if lr_label == bert_label:
                st.success(f"✅ **Both models agree:** {lr_label.upper()}")
            else:
                st.warning(f"⚠️ **Models disagree!** LR predicts '{lr_label}', BERT predicts '{bert_label}'")
        else:
            st.warning("Please enter a message")

# TAB 3: Example Tests
with tab3:
    st.header("Pre-loaded Test Examples")
    st.markdown("Click any example to see how both models classify it")
    
    examples = {
        "Clean Messages": [
            "Thanks for sharing this article, very informative!",
            "Great product, highly recommend to everyone",
            "Looking forward to the meeting tomorrow",
            "Have a wonderful day!"
        ],
        "Toxic Messages": [
            "you're an annoying idiot",
            "go kill yourself loser",
            "stupid piece of trash",
            "i hope you die"
        ],
        "Spam Messages": [
            "CLICK HERE FOR FREE MONEY!!!",
            "Win free iPhone now! Limited time offer!",
            "Work from home earn $5000 weekly",
            "Congratulations you won the lottery"
        ],
        "Adversarial (Obfuscated)": [
            "fr33 m0n3y cl1ck here",
            "y0u suck 1d10t",
            "w1n ca$h pr1ze now",
            "you're a d-u-m-b-a-s-s"
        ],
        "Platform-Specific": [
            "kys you suck (Discord)",
            "Kindly shut up (WhatsApp)",
            "you're hella embarassing ngl give up(TikTok)",
            "you fell off loser(Twitter)"
        ]
    }
    
    for category, msgs in examples.items():
        st.subheader(f"{category}")
        
        cols = st.columns(2)
        for idx, msg in enumerate(msgs):
            with cols[idx % 2]:
                if st.button(f"Test: {msg[:40]}...", key=f"{category}_{idx}", use_container_width=True):
                    st.session_state.test_message = msg
                    st.rerun()
    
    # Display results if a test message was clicked
    if 'test_message' in st.session_state:
        st.markdown("---")
        st.subheader("Test Results")
        
        msg = st.session_state.test_message
        st.info(f"**Testing:** {msg}")
        
        processed = preprocess_text(msg)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Logistic Regression**")
            X_vec = vectorizer.transform([processed])
            lr_pred = lr_model.predict(X_vec)[0]
            lr_probs_raw = lr_model.predict_proba(X_vec)[0]
            
            # Pad to 4 classes
            lr_probs = np.zeros(4)
            for i, class_idx in enumerate(lr_model.classes_):
                lr_probs[class_idx] = lr_probs_raw[i]
            
            lr_label = REVERSE_MAP[lr_pred]
            
            st.markdown(f"**Prediction:** {LABEL_EMOJIS[lr_label]} {lr_label.upper()}")
            st.markdown(f"**Confidence:** {lr_probs[lr_pred]*100:.1f}%")
        
        with col2:
            st.markdown("**DistilBERT**")
            inputs = bert_tokenizer(processed, return_tensors="pt", truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                bert_probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
                bert_pred = bert_probs.argmax()
            bert_label = REVERSE_MAP[bert_pred]
            
            st.markdown(f"**Prediction:** {LABEL_EMOJIS[bert_label]} {bert_label.upper()}")
            st.markdown(f"**Confidence:** {bert_probs[bert_pred]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Research Project:</strong> Comparative Analysis of Classical vs Transformer Models for Content Moderation</p>
    <p>Models trained on 34,575 samples from Jigsaw, SMS Spam, and Civil Comments datasets</p>
    <p style="font-size: 12px;"> Logistic Regression: 0.1ms inference | DistilBERT: 30ms inference</p>
</div>
""", unsafe_allow_html=True)