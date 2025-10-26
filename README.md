# toxic-spam-detection

# 💬 Cross-Platform Toxic & Scam Message Detection in DMs

A lightweight, real-time NLP model that detects **toxic**, **scam**, **both**, or **safe** messages across Discord, Instagram, WhatsApp, and SMS-style messages. Focused on multi-source realism, adversarial text robustness, and mobile/web deployment.

---

## 📌 Project Objectives

- Build a **multi-class classification model**: `[toxic]`, `[spam/scam]`, `[both]`, `[safe]`
- Simulate realistic DMs using platform-specific slang, emoji use, and obfuscated language
- Create a **balanced, multi-source dataset** using open data (Jigsaw, SMS Spam, Civil Comments, etc.) + simulated messages
- Evaluate lightweight models (Logistic Regression, TinyBERT, DistilBERT) for **accuracy vs inference time**
- Deploy an interactive **Streamlit app** with interpretability tools (LIME/SHAP)

---

## 🧠 Research Scope

- NLP + Social Media Safety
- Lightweight transformer-based toxic/scam detection
- Cross-platform generalization and domain adaptation
- Real-time use case simulation and adversarial robustness

---

## 🗂️ Directory Structure
├── data/ # Datasets (CSV, raw + simulated)
├── notebooks/ # Exploratory and model training notebooks
├── src/ # Python modules: preprocessing, modeling, evaluation
├── streamlit_app/ # Code for UI and deployment
├── requirements.txt # Project dependencies
├── README.md
└── setup_colab.ipynb # Optional colab bootstrapping

---

## 📊 Datasets Used

| Dataset | Source | Label Types | Notes |
|--------|--------|-------------|-------|
| Jigsaw Toxic Comments | Kaggle | Toxicity | Balanced + cleaned |
| SMS Spam | UCI | Spam, Safe | Formal, short |
| Civil Comments | Kaggle | Toxicity | More nuanced |
| HateXplain | HuggingFace | Toxicity, Rationale | Includes explanations |
| Simulated DMs | Custom | All 4 classes | Discord/IG/WhatsApp formats |

---

## 🛠️ Models

| Model | Type | Notes |
|-------|------|-------|
| TF-IDF + Logistic Regression | Classical baseline | Fast, interpretable |
| TinyBERT | Transformer | Lightweight, good for mobile |
| DistilBERT | Transformer | Tradeoff between speed and accuracy |
| LSTM + GloVe | Deep learning | Benchmarked historically |
| Residual LSTM + Transfer | Research inspired | For cross-domain generalization |

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1 (macro + per class)
- Confusion Matrix
- ROC-AUC (optional)
- Inference Time (ms per message)
- LIME/SHAP visual explanations

---

## 🧪 Try It Out (Coming Soon)

A Streamlit UI where users can:
- Enter a message
- View prediction (toxic, scam, both, safe)
- See a confidence score and word-level heatmap
- Measure latency in real-time

---

## 🔍 Research Proposal & Docs

- Literature Review (5 key papers)
- Methodology & Research Objectives
- Dataset Inventory Table
- Model Comparison Table
- Streamlit Walkthrough

📝 *Available in `docs/` and project report (coming soon)*

---

## 📌 Future Work

- Domain adaptation via fine-tuning
- Multilingual support (Hinglish, Spanglish)
- DM thread-level classification
- Real-world deployment optimization

---

## 🙋‍♀️ Author & Mentorship

**Author**: Nithika Neela  
**Counselor/Advisor**: Rishika Porandla 
**Looking for academic mentorship** → contact me via email or GitHub

---

## 📜 License

This project is licensed under the MIT License.
