"""
cd data_process
python pipeline/component_negation_converter.py
"""

import spacy
import pandas as pd

# from profanity_check import predict, predict_prob
# from better_profanity import profanity
from googleapiclient import discovery


from spacy import displacy
from pathlib import Path
from tqdm import tqdm

from pipeline.pipeline_component import PipelineComponent

import logging
import os
import time

logger = logging.getLogger(__name__)

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API")


class ContentFilter(PipelineComponent):
    description = "Filter out toxic records from the knowledge bank"
    config_layer = "content_filter"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df

    def run(self):
        df = self.read_input()
        fields = [
            "cultural group",
            "context",
            "goal",
            "relation",
            "actor",
            "recipient",
            "actor's behavior",
            "recipient's behavior",
            "other descriptions",
            "topic",
        ]
        attributes = [
            "TOXICITY",
            "PROFANITY",
            "INSULT",
            "IDENTITY_ATTACK",
            "THREAT",
            "SEVERE_TOXICITY",
        ]
        attributes_scores = {att: [] for att in attributes}
        for i in tqdm(range(df.shape[0])):
            sent = ", ".join(
                f"{df.iloc[i][field]}"
                for field in fields
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
        self.save_output(df)

    def save_output(self, df):
        logger.info(f"save to {self._local_config['output_file']}")
        df.to_csv(
            self._local_config["output_file"],
            index=False,
        )


if __name__ == "__main__":
    pass
