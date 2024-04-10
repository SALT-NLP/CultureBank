# CultureBank
the code is adapted from https://github.com/cultural-csk/candle

# CultureBank

## training small classifiers (cultural relevance, and controversial classifier)
1. if your data need some custimization
python /sailhome/weiyans/CultureBank/data_process/pipeline/culture_relevance_classification/scripts/train_culture_relevance_classifier.py
2. distill the summarizer? (ryan? i am not sure if we need to add this code)

# pipeline
## add some dumb file
# /sailhome/weiyans/culturebank_public/CultureBank/data_process_pipeline/dummy_data/fields.csv
CultureBank pipeline execution
1.1 running cultural relevenace classifier (provide one model, the reddit one?)
1.2 extraction (provide the fine-tuned mixtral model, the reddit one?) #ryan
2.1 clustering (with default parameters, TODO: need to add time range)
2.2 cluster summarization (provide the fine-tuned mixtral model, the reddit one?) #ryan
2.3 topic normalization? #ryan
3.1 agreement calculator 
3.2 content moderation (perspective + provide the controversial one for tiktok + keywords)
3.3 final touch, PII information

# evaluation code (Ryan)
1. generate the scenario
2. direction eval
3. grounded eval

# fine-tuning code (Ryan)
0. finetuning
1. cultureNLI eval
2. world value eval


