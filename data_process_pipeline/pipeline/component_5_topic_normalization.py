import spacy
import pandas as pd
import torch
import json
import os
from collections import Counter

from spacy import displacy
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
import math
import re
import openai
from openai import OpenAI
import asyncio
from sklearn.metrics import classification_report

from pipeline.pipeline_component import PipelineComponent
from utils.constants import CULTURAL_TOPICS
from utils.prompt_utils import TOPIC_SYSTEM_MESSAGE, TOPIC_USER_MESSAGE_TEMPLATE
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

import nltk
from nltk.stem import WordNetLemmatizer

from utils.clustering import (
    hac_clustering,
    hac_clustering_retain_index,
    secondary_clustering,
)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
nltk.download("wordnet")
logger = logging.getLogger(__name__)


class TopicNormalizer(PipelineComponent):
    description = "normalizing the topics and cultural groups"
    config_layer = "5_topic_normalizer"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]
        self._override_config()
        self._condition = f"group={self._local_config['cultural_group_threshold']}"
        self._create_new_output_dir()
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])
        self.scores = {"cluster_silhouette_score": []}
        # setup models
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"using {self.device}")
        self.sbert = SentenceTransformer(self._local_config["sbert"]["model"]).to(
            self.device
        )

    def _override_config(self):
        for key in ["cultural_group_threshold"]:
            if key in self._config and self._config[key] is not None:
                self._local_config[key] = self._config[key]

    def _create_new_output_dir(self):
        import pathlib

        new_output_dir = "/".join(
            self._local_config["output_file"].split("/")[:-1] + [self._condition]
        )
        pathlib.Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        for key in ["output_file", "output_score_file"]:
            self._local_config[key] = "/".join(
                self._local_config[key].split("/")[:-1]
                + [self._condition]
                + self._local_config[key].split("/")[-1:],
            )

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df

    def run(self):
        df = self.read_input()
        logger.info(f"total number of samples: {len(df)}")
        group_clusters = self.cultural_group_normalization(df)
        df = self.select_representative_summarization(
            df, "cultural group", group_clusters
        )

        df = self.topic_normalization(df)

        self.save_output(df)
        logger.info("Normalization Done!")

    def save_output(self, df):
        logger.info(f"save to {self._local_config['output_file']}")
        df.to_csv(
            self._local_config["output_file"],
            index=False,
        )
        with open(self._local_config["output_score_file"], "w") as fh:
            json.dump(self.scores, fh)

    def cultural_group_normalization(self, df):
        sents = [f"{df.iloc[idx]['cultural group']}" for idx, row in df.iterrows()]
        logger.info(f"this many culture groups: {len(sents)}")

        embeddings = self.sbert.encode(sents, show_progress_bar=True)
        raw_clusters, score = hac_clustering_retain_index(
            sents, embeddings, self._local_config["cultural_group_threshold"]
        )
        logger.info(f"there are a total of {len(raw_clusters)} cultural groups")
        logger.info(
            f"the size of the largest cultural group is: {max([len(cluster) for cluster in raw_clusters])}"
        )
        logger.info(
            f"the silhouette_score for the cultural group clustering is {score}"
        )
        self.scores["cultural_group_silhouette_score"] = score

        return raw_clusters

    def topic_normalization(self, df):
        df["representative_topic"] = ""
        model = self._local_config["openai"]["model"]
        temperature = self._local_config["openai"]["temperature"]
        max_tokens = self._local_config["openai"]["max_tokens"]
        top_p = self._local_config["openai"]["top_p"]
        seed = self._local_config["openai"]["seed"]

        for idx, _ in tqdm(df.iterrows(), total=len(df)):
            for _ in range(10):
                try:
                    df_line = df.iloc[idx]
                    system_message = TOPIC_SYSTEM_MESSAGE.format(CULTURAL_TOPICS)
                    user_message = TOPIC_USER_MESSAGE_TEMPLATE.format(df_line["topic"])
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ]

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        seed=seed,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    response_content = response.choices[0].message.content
                    prompt_tokens = response.usage.prompt_tokens

                    summarized_topic = response_content.strip()
                    summarized_topic = re.sub(r'[\'"]', "", summarized_topic)
                    summarized_topic = re.sub(r"\.$", "", summarized_topic)
                    if summarized_topic not in CULTURAL_TOPICS:
                        print(
                            f"row {idx}: the summarized topic {summarized_topic} does not fit into any of the predefined themes, retrying..."
                        )
                        continue
                    df.at[idx, "representative_topic"] = summarized_topic
                    break
                except Exception as e:
                    print(f"encountered error at row {idx}: {e}")
                    print("retrying...")
            continue
        return df

    @staticmethod
    def select_representative_summarization(
        df, cluster_target, raw_clusters, strategy="majority"
    ):
        final_values = [None] * df.shape[0]
        final_values_count = [None] * df.shape[0]
        final_cluster_id = [None] * df.shape[0]
        for i, cluster in enumerate(tqdm(raw_clusters)):
            actual_values = []
            for idx, _ in cluster:
                row = df.iloc[idx]
                if cluster_target == "topic":
                    actual_values.append(row["representative_topic"])
                else:
                    actual_values.append(row[cluster_target])
            if strategy == "majority":
                # Count the occurrences of each element
                vote_counts = Counter(actual_values)
                # Find the majority vote
                majority_vote, majority_count = vote_counts.most_common(1)[0]
                rep_topic = majority_vote
            else:
                raise NotImplementedError

            for idx, _ in cluster:
                final_values[idx] = rep_topic
                if strategy == "majority":
                    final_values_count[idx] = majority_count
                final_cluster_id[idx] = i
        df[f"representative_{cluster_target}"] = final_values
        if strategy == "majority":
            df[f"representative_{cluster_target}_count"] = final_values_count
        df[f"representative_{cluster_target}_cluster_id"] = final_cluster_id
        return df
