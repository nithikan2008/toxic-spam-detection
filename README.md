# 🛡️ When Classical ML Outperforms Transformers: Content Moderation Research

**A comparative robustness analysis of classical machine learning vs transformer models for social media content moderation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Findings

This research challenges the assumption that transformer models universally outperform classical ML for text classification. Through comprehensive robustness testing across adversarial attacks and domain shifts:

- **Logistic Regression achieved 2x better adversarial robustness** (37.4% vs 68.9% performance degradation)
- **LR obtained higher macro-F1 on clean data** (0.815 vs 0.712) despite transformers' benchmark dominance
- **DistilBERT showed slight advantages in cross-platform generalization** (50.7% vs 53.9% degradation)
- **LR offers 300x faster inference** (0.10ms vs 30.70ms per message)

**Conclusion:** For production content moderation systems facing adversarial users (spammers, trolls who deliberately evade detection), carefully tuned classical models offer superior cost-benefit profiles.

---

## 📊 Quick Results Overview

### Performance on Clean Data
| Model | Macro-F1 | Accuracy | Inference Speed |
|-------|----------|----------|-----------------|
| **Logistic Regression** | **0.815** | 92.8% | 0.10 ms/msg |
| DistilBERT | 0.712 | **94.9%** | 30.70 ms/msg |

### Adversarial Robustness (Character-Level Obfuscation)
| Model | Clean F1 | Adversarial F1 | Performance Drop |
|-------|----------|----------------|------------------|
| **Logistic Regression** | 0.815 | 0.511 | **37.4%** ✅ |
| DistilBERT | 0.712 | 0.221 | 68.9% |

### Cross-Platform Generalization
| Platform | LR F1 | BERT F1 | Best Model |
|----------|-------|---------|------------|
| Discord (Gaming) | 0.282 | **0.310** | BERT ✅ |
| WhatsApp (Formal) | **0.462** | 0.379 | LR ✅ |
| TikTok/IG (Gen Z) | 0.384 | **0.364** | BERT ✅ |

---

## 🔬 Research Contributions

1. **Dual-Axis Robustness Evaluation**: First study to systematically compare adversarial attack resistance AND cross-domain generalization for content moderation
2. **Architectural Vulnerability Analysis**: Identified that transformer subword tokenization creates systematic blind spots for character-level perturbations
3. **Production Deployment Insights**: Practical guidance for choosing models based on threat model (adversarial vs domain shift)
4. **Adversarial Test Suite**: Curated 50 obfuscated messages simulating real evasion tactics (leetspeak, spacing, emojis)
5. **Platform-Specific Benchmarks**: Created test sets for Discord, WhatsApp, and TikTok/Instagram communication styles

---

## 🗂️ Project Structure

```
toxic-spam-detection/
├── src/
│   ├── train.py                    # Model training pipeline
│   ├── preprocessing.py            # Text normalization pipeline
│   ├── adversarial_test.py         # Adversarial robustness evaluation
│   ├── cross_platform_test.py      # Domain adaptation testing
│   ├── models.py                   # Model configurations
│   └── app.py                      # Streamlit demo application
├── data/
│   └── README.md                   # Dataset download instructions
├── results/
│   ├── adversarial_robustness_comparison.png
│   ├── cross_platform_analysis.png
│   ├── adversarial_results.csv
│   └── cross_platform_results.csv
├── saved_models/
│   └── distilbert/                 # Fine-tuned DistilBERT checkpoint
├── requirements.txt
├── logbook.md                      # Research process documentation
└── README.md
```

---

## 📚 Datasets

**Training Data (34,575 samples from 5 sources):**

| Dataset | Source | Size | Label Types | Notes |
|---------|--------|------|-------------|-------|
| Jigsaw Toxic Comments | Kaggle | 20,000 | Toxicity | Reddit discussions |
| YouTube Toxic Comments | Kaggle | 8,000 | Toxicity | Video comment threads |
| The Toxicity Dataset | GitHub | 3,575 | Toxicity | Multi-platform, implicit toxicity |
| SMS Spam Collection | UCI ML Repository | 5,574 | Spam/Safe | Mobile marketing, phishing |
| Email Spam Dataset | Kaggle | 2,426 | Spam/Safe | Professional email scams |

**Test Sets:**
- Standard test set: 6,915 samples (20% stratified split)
- Adversarial test set: 50 manually curated obfuscated messages
- Cross-platform tests: Discord (31), WhatsApp (28), TikTok/IG (29)

**Download Instructions:** See `data/README.md`

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nithikan2008/toxic-spam-detection.git
cd toxic-spam-detection

# Install dependencies
pip install -r requirements.txt

# Download datasets (see data/README.md for links)
```

### Run Experiments

```bash
# Train models
python src/train.py

# Adversarial robustness testing
python src/adversarial_test.py

# Cross-platform evaluation
python src/cross_platform_test.py
```

### Interactive Demo

```bash
# Launch Streamlit app
streamlit run src/app.py
```

The demo provides:
- Single message classification with confidence scores
- Side-by-side model comparison
- Pre-loaded test examples across categories
- Real-time inference latency measurement

---

## 🛠️ Models Evaluated

### Classical Baseline
**Logistic Regression with TF-IDF**
- Vectorization: TF-IDF with character + word bigrams (max 20k features)
- Regularization: L2 penalty, class-weighted for imbalance
- Training time: ~2 minutes
- Inference: 0.10 ms/message

### Transformer Model
**Fine-tuned DistilBERT**
- Architecture: distilbert-base-uncased (66M parameters)
- Fine-tuning: 2 epochs on 5,000 samples (computational constraint)
- Batch size: 8, Learning rate: 2e-5
- Training time: ~35 minutes (CPU)
- Inference: 30.70 ms/message

---

## 📈 Evaluation Methodology

### Metrics
- **Macro-F1**: Primary metric (unweighted average across classes)
- **Per-class Precision/Recall**: Minority class performance tracking
- **Confusion Matrices**: Error pattern analysis
- **Inference Latency**: Real-time deployment feasibility

### Robustness Testing

**Adversarial Attacks:**
1. Leetspeak substitutions: "fr33 m0n3y", "y0u 1d10t"
2. Strategic spacing: "f.r.e.e m.o.n.e.y", "y o u  s u c k"
3. Emoji substitution: "💰💰💰 click here", "🤬🤬🤬"
4. Clean controls: Verify no false degradation

**Cross-Platform Transfer:**
- Train on Reddit/SMS (formal text)
- Test on Discord (gaming slang), WhatsApp (international), TikTok/IG (Gen Z)
- Zero-shot evaluation (no platform-specific fine-tuning)

---

## 🔍 Key Technical Insights

### Why Does Classical ML Win on Adversarial Robustness?

**1. Tokenization Brittleness**
- DistilBERT's WordPiece tokenizer fragments obfuscated words into out-of-vocabulary tokens
- "money" (single token) → "m0n3y" → ["m", "##0", "##n", "##3", "##y"] (5 weak tokens)
- Character n-grams naturally capture overlapping substrings: "m0n", "0n3" persist

**2. Explicit Feature Engineering**
- TF-IDF bigrams like "free_money" and "click_here" explicitly encode spam patterns
- Partial matches ("fr33_money") still trigger related n-grams (graceful degradation)

**3. Sample Efficiency**
- 66M parameters trained on 5K samples → memorization risk
- 20K TF-IDF features with L2 regularization → better generalization

### When Transformers Show Advantages

**Cross-Platform Generalization:**
- Novel slang ("no cap", "hardstuck silver") benefits from contextual embeddings
- Domain shift (gaming → formal) where vocabulary changes but semantics remain
- Only 3.2 percentage points improvement—marginal given 300x slower inference

---

## 📄 Research Paper

**Full paper available:** [Link to paper PDF]

**Citation:**
```bibtex
@article{neela2024classical,
  title={When Classical Machine Learning Outperforms Transformers: A Robustness Analysis of Content Moderation Systems},
  author={Neela, Nithika},
  year={2024},
  note={Independent research project}
}
```

---

## 🎥 Demo Video

[Link to demo video showing Streamlit app]

---

## 🔮 Future Work

### Immediate Extensions
- **Full-dataset DistilBERT training** (27K samples vs current 5K)
- **Adversarial training** incorporating obfuscated examples during fine-tuning
- **Character-aware transformers** (CharacterBERT) for tokenization robustness
- **Ensemble methods** combining LR robustness with BERT contextual understanding

### Advanced Research Directions
- Semantic-preserving adversarial attacks (paraphrasing, sarcasm)
- Multilingual evaluation (code-switching evasion tactics)
- Real-world deployment study with live user feedback
- Automated adversarial generation at scale

---

## 📖 Documentation

- **Research Process:** See `logbook.md` for detailed development timeline
- **Literature Review:** See paper Section 2 (Related Work)
- **Preprocessing Details:** See paper Section 3.2
- **Full Results:** See `results/` folder for all figures and CSV tables

---

## 🤝 Contributing

This is an independent research project, but feedback and suggestions are welcome! Feel free to:
- Open issues for bugs or questions
- Suggest additional experiments or datasets
- Share deployment experiences

---

## 👩‍🔬 Author

**Nithika Neela**  
High School Independent Researcher  
📧 [Your Email]  
🔗 [LinkedIn/Personal Website]

**Advisor:** Rishika Porandla

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets:** Jigsaw/Kaggle, UCI ML Repository, GitHub Toxicity Dataset
- **Libraries:** HuggingFace Transformers, scikit-learn, Streamlit
- **Inspiration:** Research on adversarial NLP robustness and domain adaptation

---

## 📊 Project Stats

- **Lines of Code:** ~2,000+
- **Training Samples:** 34,575
- **Test Samples:** 7,053 (standard + adversarial + cross-platform)
- **Models Trained:** 2 (LR, DistilBERT)
- **Experiments Run:** 6 (clean, adversarial, 3 cross-platform splits)
- **Development Time:** 3 months (September - November 2024)

---

**⭐ If this research helped you, please star the repository!**
