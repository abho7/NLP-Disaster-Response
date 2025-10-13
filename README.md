# Harnessing Natural Language Processing for Disaster Response and Crisis Management

This repository contains all code, documentation, and configuration files used in the study  
**"Harnessing Natural Language Processing for Disaster Response and Crisis Management"** submitted to the *National High School Journal of Science (NHSJS), 2025.*

---

## 📘 Contents
- `data/` — Scripts for dataset preprocessing and annotation schema  
- `models/` — Fine-tuning scripts and configuration files for BERT  
- `evaluation/` — Code for performance metrics, error analysis, and ROC curve generation  
- `figures/` — Original figures used in the manuscript (e.g., ROC curves, tables)  
- `requirements.txt` — Dependencies for reproducibility  

---

## ⚙️ Reproduction Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbhineethDuddela/NLP-Disaster-Response.git
   cd NLP-Disaster-Response

 pip install -r requirements.txt

 python models/train_bert.py

 python evaluation/evaluate.py

