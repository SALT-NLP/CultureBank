"""
This file will output two files, one with all the datapoints >= hard_api_threshold filtered, and one file with nuanced problematic ones to annotate 
"""

import spacy
import pandas as pd

# from profanity_check import predict, predict_prob
# from better_profanity import profanity
from googleapiclient import discovery
from utils.constants import CULTUREBANK_FIELDS
from presidio_analyzer import AnalyzerEngine
from transformers import pipeline
import sys


from spacy import displacy
from pathlib import Path
from tqdm import tqdm

from pipeline.pipeline_component import PipelineComponent

import logging
import os
import time

logger = logging.getLogger(__name__)

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API")


class ContentModeration(PipelineComponent):
    description = "Content moderation for the knowledge bank, select records for manual annotation"
    config_layer = "7_content_moderation"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])

        # hard threshold, anything above or equal will be removed without annotation
        self.hard_api_threshold = self._local_config["hard_api_threshold"]
        self.soft_api_threshold = self._local_config["soft_api_threshold"]
        self.fields_to_check = [
            field for field in CULTUREBANK_FIELDS if field != "norm"
        ]

        # perspective api
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        # our own controversial classifier
        self.classifier = self.load_classifier()

    def load_classifier(self):
        # load_model
        classifier = pipeline(
            "text-classification",
            model=self._local_config["model_dir"],
            device=self._local_config["device"],
        )
        return classifier

    def read_keyword_list(self):
        # get controversial keywords
        with open(self._local_config["keyword_list_dir"]) as fh:
            words = fh.readlines()
            words = [" " + w.strip().strip("\n") + " " for w in words]
        return words

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df

    def run(self):
        df = self.read_input()
        # get perspective scores
        df = self.get_perspective_scores(df)

        # hard filter by perspective score, anything that belongs to [hard, 1]
        ids_with_api_hard = self.filter_by_perspective_api(df, self.hard_api_threshold)
        df = df[~df.cluster_id.isin(ids_with_api_hard)]
        self.save_output(df, save_dir=self._local_config["output_file"])

        # now get the nuanced cases for manual annotation
        # 1. predicted by our classifier
        df = self.predict_by_classifier(df)
        ids_with_predicted = df[df.pred_label == "controversial"].cluster_id.tolist()
        # 2. detect pii information
        df = self.detect_pii(df)
        ids_with_pii = self.filter_by_pii(df)
        # 3. keyword filtering
        ids_with_keywords = self.filter_by_keywords(df)
        # 4. above the perspective soft threshold, anything that belongs to [soft, hard)
        ids_with_api = self.filter_by_perspective_api(df, self.soft_api_threshold)
        final_ids = list(
            set(ids_with_predicted + ids_with_keywords + ids_with_api + ids_with_pii)
        )
        controversial_df = df[df.cluster_id.isin(final_ids)]
        controversial_df = controversial_df.sample(frac=1, random_state=42)
        controversial_df[self._local_config["controversial_field_name_to_annotate"]] = (
            None
        )
        logger.info(f"this many for annotation {controversial_df.shape[0]}\n")
        logger.info(
            f"PLEASE ANNOTATE THEM NOW!! Type OK to continue {self._local_config['output_file_for_manual_annotation']}"
        )
        user_input = input()
        if user_input == "OK":
            pass
        else:
            logger.info(f"you didn't type OK! please annotate the file above")
        self.save_output(
            controversial_df,
            save_dir=self._local_config["output_file_for_manual_annotation"],
        )

    def get_perspective_scores(self, df):
        attributes = [
            "TOXICITY",
            "PROFANITY",
            "INSULT",
            "IDENTITY_ATTACK",
            "THREAT",
            "SEVERE_TOXICITY",
        ]  # everything perpective api has as of 2024/03
        attributes_scores = {att: [] for att in attributes}
        for i in tqdm(range(df.shape[0])):
            sent = ", ".join(
                f"{df.iloc[i][field]}"
                for field in self.fields_to_check
                if pd.notna(df.iloc[i][field])
            )
            analyze_request = {
                "comment": {"text": sent},
                "languages": ["en"],
                "requestedAttributes": {
                    "TOXICITY": {},
                    "PROFANITY": {},
                    "INSULT": {},
                    "IDENTITY_ATTACK": {},
                    "THREAT": {},
                    "SEVERE_TOXICITY": {},
                },
                "doNotStore": True,
            }
            start_time = time.time()
            response = self.client.comments().analyze(body=analyze_request).execute()
            # response = json.dumps(response, indent=2)
            for attribute in attributes:
                score = response["attributeScores"][attribute]["summaryScore"]["value"]
                if attribute in response["attributeScores"]:
                    attributes_scores[attribute].append(score)
                else:
                    attributes_scores[attribute].append(None)
            elapsed_time = time.time() - start_time
            if elapsed_time < 1.2:
                time.sleep(1.2 - elapsed_time)
        for attribute in attributes:
            df[attribute] = attributes_scores[attribute]

        return df

    def filter_by_keywords(self, df):
        # get the ids identified by keywords
        ids_with_keywords = []
        keywords = self.read_keyword_list()
        for i in range(df.shape[0]):
            line = df.iloc[i]
            for field in self.fields_to_check:
                if type(line[field]) is str:
                    for w in keywords:
                        if w.lower() in line[field].lower():
                            ids_with_keywords.append(line["cluster_id"])
        ids_with_keywords = sorted(list(set(ids_with_keywords)))
        logger.info(f"this keywords {len(ids_with_keywords)}")
        return ids_with_keywords

    def filter_by_perspective_api(self, df, threshold):
        # get the ids identified by api
        ids_with_api = df[
            (df["TOXICITY"] >= threshold)
            | (df["PROFANITY"] >= threshold)
            | (df["INSULT"] >= threshold)
            | (df["IDENTITY_ATTACK"] >= threshold)
            | (df["THREAT"] >= threshold)
            | (df["SEVERE_TOXICITY"] >= threshold)
        ].cluster_id.tolist()
        logger.info(f"this many api {len(ids_with_api)}")
        return ids_with_api

    def detect_pii(self, df):
        analyzer = AnalyzerEngine()
        entities_to_check = analyzer.get_supported_entities()
        entities_to_check = [
            item
            for item in entities_to_check
            if item not in ["NRP", "LOCATION", "DATE_TIME"]
        ]
        PII_list = []
        concat_list = []
        keywords_list = []
        for i in tqdm(range(df.shape[0])):
            df_line = df.iloc[i][self.fields_to_check]
            concatenated_texts = ", ".join(
                [str(item) for item in df_line if pd.notna(item)]
            )
            concat_list.append(concatenated_texts)
            results = analyzer.analyze(
                text=concatenated_texts, entities=entities_to_check, language="en"
            )
            keywords = {
                concatenated_texts[result.start : result.end]: result.entity_type
                for result in results
            }
            keywords_list.append(keywords)
            if len(results) == 0:
                PII_list.append(None)
            else:
                PII_list.append(results)
        df["pii_result"] = PII_list
        df["concat_list"] = concat_list
        df["keywords_list"] = keywords_list
        return df

    def filter_by_pii(self, df):
        ids_with_pii = df[df.keywords_list != {}].cluster_id.tolist()
        logger.info(f"this many PII {len(ids_with_pii)}")
        return ids_with_pii

    def predict_by_classifier(self, df):
        # run classifier to predict controversial data

        id2label = {0: "non-controversial", 1: "controversial"}

        df["text_for_controversial_prediction"] = df.apply(
            lambda x: ", ".join(
                [x[field] for field in self.fields_to_check if not pd.isnull(x[field])]
            ),
            axis=1,
        )
        test_pred = self.classifier(
            [str(r) for r in df["text_for_controversial_prediction"].tolist()],
            batch_size=512,
        )
        df["pred_label"] = [id2label[int(pred["label"][-1])] for pred in test_pred]
        df["pred_score"] = [round(pred["score"], 2) for pred in test_pred]
        return df

    def save_output(self, df, save_dir):
        logger.info(f"save to {save_dir}")
        df.to_csv(
            save_dir,
            index=False,
        )
