# Harnessing Natural Language Processing for Disaster Response and Crisis Management

This repository contains the complete source code, model configurations, and documentation for the study  
**"Harnessing Natural Language Processing for Disaster Response and Crisis Management"**  
submitted to the *National High School Journal of Science (NHSJS), 2025.*

---

## 📖 Overview

This project demonstrates how Natural Language Processing (NLP) can be applied to classify disaster-related tweets by urgency.  
It includes:

- Fine-tuning **BERT** for binary urgency classification (`urgent` vs `not urgent`).  
- Evaluation using **classification metrics**, confusion matrices, and ROC curves.  
- Sample data to illustrate the expected schema; full datasets can be re-scraped or provided upon request.  

All code is lightweight and designed for **reproducibility** and review.

---

## 📁 Repository Structure

NLP-Disaster-Response/
├── data/         # Sample CSV data and preprocessing scripts
├── models/       # Model training scripts (BERT and Naive Bayes)
├── evaluation/   # Model evaluation and ROC curve generation
├── figures/      # Publication figures (ROC curves, tables, etc.)
├── results/      # Output metrics and model checkpoints
└── requirements.txt  # Python dependencies


---

## ⚙️ Reproduction Instructions

### Clone the repository
```bash
git clone https://github.com/AbhineethDuddela/NLP-Disaster-Response.git
cd NLP-Disaster-Response
pip install -r requirements.txt
python models/train_bert.py
python evaluation/evaluate.py

## NOTE
Original tweets are not included due to Twitter policy. Scripts to re-scrape and annotate are provided.

The sample_tweets.csv file illustrates the CSV format expected by the training and evaluation scripts.

To reproduce figures from the paper, run python evaluation/evaluate.py.

The repository is structured to make it easy for reviewers and other researchers to reproduce the results.
