# latent-themes-LLMs
Code and data for ICWSM 2025 paper "Discovering Latent Themes in Social Media Messaging: A Machine-in-the-Loop Approach Integrating LLMs".

## Necessary Libraries
The model is implemented in Python 3. 

Other major libraries needed:

Python 3.9.19

PyTorch 1.13.1

openai

jupiter notebook

pandas

gensim

nltk

nltk.tag

spacy

emoji

sklearn

scipy

matplotlib

numpy

preprocessor

transformers

seaborn

BERTopic

## Instructions for running code:

For Coverage and evaluation of clusters on COVID-19 (Table 2):

For pre-existing theme: 
evaluate_cluster_pre_existing_theme_covid.py

This will give covered ads based on 0.5 thresholding on distance pre_existing_covid.csv
This will give uncover ads based on 0.5 thresholding on distance uncover_0.5_covid.csv

Getting top-k best assigned ads using pre-existing themes covid:
Run Best_assigned_ads_pre_existing_theme_covid.ipynb


Do clustering. After clustering and refining cluster for iteration 1, do evaluation of clusters in iteration 1 and this will give covid_thm_iter1.csv file.

python evaluate_cluster_theme_iter1_covid.py
 
Do clustering on uncover ads. Then, after clustering and refining cluster For iteration 2, do evaluation of clusters in iteration 2 and this will give covid_thm_iter2.csv file.

python evaluate_cluster_theme_iter2_covid.py


For mapping ads->themes using LLMs mapper, run following codes:

python ChatGPT_summary_text_match_climate.py
python ChatGPT_summary_text_match_covid.py


For Mapping Quality and Qualitative Analysis for COVID-19 (Figure 7, Figure 8, Figure 9, Table 4):

Run qualitative_analysis_mapping_covid.ipynb

