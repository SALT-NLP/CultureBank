# CultureBank
Quick Links: [[dataset-tiktok]](https://huggingface.co/datasets/SALT-NLP/CultureBank-TikTok) [[dataset-reddit]](https://huggingface.co/datasets/SALT-NLP/CultureBank-Reddit) [[Models](https://huggingface.co/datasets/SALT-NLP/CultureBank-Reddit)] [[Project Page]](https://salt-nlp.github.io/Design2Code/) [[Paper]](https://salt-nlp.github.io/Design2Code/)

#todo:
# PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API")

## data_process_pipeline
## add some dumb file
# /sailhome/weiyans/culturebank_public/CultureBank/data_process_pipeline/dummy_data/comments.csv
CultureBank pipeline execution
1.1 running cultural relevenace classifier (provide one model, the reddit one?)
1.2 extraction (provide the fine-tuned mixtral model, the reddit one?) #ryan
2.1 clustering (with default parameters, TODO: need to add time range)
2.2 cluster summarization (provide the fine-tuned mixtral model, the reddit one?) #ryan
2.3 topic normalization? #ryan
3.1 agreement calculator 
3.2 content moderation (perspective + provide the controversial one for tiktok + keywords)
3.3 final touch, PII information

## evaluation code (Ryan)
1. generate the scenario (another adapter)
2. direction eval
3. grounded eval

## fine-tuning code (Ryan)
0. finetuning
1. cultureNLI eval
2. world value eval

# released model
1. extracgor
2. summarizer
3. question generator
4. culturally aware llm. 
CultureBank-extracgor
CultureBank-Summarizer

## Acknowledgement

The codebase is adapted from [Candle](https://github.com/cultural-csk/candle) ([paper](https://arxiv.org/abs/2210.07763)) which is [under this license](https://github.com/cultural-csk/candle?tab=CC-BY-4.0-1-ov-file). Thanks for the amazing work!

If you find our work helpful, please consider citing our paper:

```
@misc{si2024design2code,
    title={CultureBank: An Online Community-Driven Knowledge Base Towards Culturally Aware Language Technologies},
    author={Weiyan Shi and Ryan Li and Yutong Zhang and Ruibo Liu and Diyi Yang},
    year={2024},
    eprint={2403.03163},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

We welcom all kinds of contributions. If you have any questions, feel free to leave issues or email us.

