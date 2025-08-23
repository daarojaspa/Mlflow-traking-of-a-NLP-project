# Text tickets clasificator

## Overview
This project explores an NLP pipeline for support ticket classification.  
The goal is to build a reproducible workflow that includes:  
- Data preprocessing and feature extraction.  
- Experiment tracking using MLflow.  
- Baseline model comparison to evaluate performance before moving to production.  

## Repository Structure

### Experiments branch

- **introMlflow.ipynb** → Example usage of MLflow tracking URIs and logging.  
- **utils/**
  - `textProcessing.py`: Preprocessing (tokenization, stopword removal, stemming, POS tagging).  
  - `featureExtraction.py`: Feature engineering for downstream tasks.  
- **tracking/**
  - `data_raw/`: Original dataset (JSON + first exploration notebook).  
  - `data_processed/`: Cleaned and labeled dataset, plus vectorizer assets.  
  - `base_line.ipynb`: Baseline ML models with metrics comparison.  

### Orchestraition

At this point i have found  disconcordance between data in experiments and data here, i whent back and relize that: i wasn using the wrong data set in base line experiments, and that the splitting of the data made   mandatory the use of the same artefacts in trainning  and production, but the data leakage prevention was making  the  way i was storing data messi, so i took this oportunity  to impruve the data labeling process and storage,  putting it in a separate flow, and implementing  pre train models and bert topics to have better labels. i will let the  experiment branch for educational and comparative porpusess

- **orchestration/**: Scripts to run the pipeline end-to-end.  

## Usage
1. Run preprocessing:  
   ```bash
   python utils/textProcessing.py
## Data