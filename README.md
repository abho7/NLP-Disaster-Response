# Harnessing Natural Language Processing for Disaster Response and Crisis Management

This repository contains the complete source code, model configurations, and documentation for the study  
**"Harnessing Natural Language Processing for Disaster Response and Crisis Management"**  
submitted to the *National High School Journal of Science (NHSJS), 2025.*

---

## 📁 Repository Structure
NLP-Disaster-Response/
│
├── data/                # Data preprocessing scripts and annotation schema
├── models/              # Model training scripts (BERT and Naive Bayes)
├── evaluation/          # Model evaluation and ROC curve generation
├── figures/             # Publication figures (ROC curves, tables, etc.)
├── results/             # Output metrics and model checkpoints
└── requirements.txt     # Dependencies

---

## ⚙️ Reproduction Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AbhineethDuddela/NLP-Disaster-Response.git
cd NLP-Disaster-Response
pip install -r requirements.txt
python models/train_bert.py
python evaluation/evaluate.py
