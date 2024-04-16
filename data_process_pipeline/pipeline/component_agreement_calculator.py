import spacy
import pandas as pd

from spacy import displacy
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from utils.util import parse_to_int

from pipeline.pipeline_component import PipelineComponent
from utils.constants import CULTUREBANK_FIELDS


import logging

logger = logging.getLogger(__name__)


class AgreementCalculatorForReddits(PipelineComponent):
    description = "Gather the summarization and calculate the agreement among comments"
    config_layer = "agreement_calculator"

    def __init__(self, config: dict):
        super().__init__(config)
        # get local config
        self._local_config = config[self.config_layer]
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])

    def read_input(self):
        df_before_cluster = pd.read_csv(
            self._local_config["original_before_cluster_file"]
        )
        df_cluster = pd.read_csv(self._local_config["cluster_file"])
        df_summary = pd.read_csv(self._local_config["cluster_summarization_input_file"])
        if self._config["dry_run"] is not None:
            df_summary = df_summary.iloc[: self._config["dry_run"]]
        return df_before_cluster, df_cluster, df_summary

    def run(self):
        df_before_cluster, df_cluster, df_summary = self.read_input()

        df_summary = pd.merge(
            df_cluster[["cluster_id", "raw_sample_vids"]],
            df_summary,
            on="cluster_id",
            how="inner",
        )

        norm_total_list = []
        norm_confidence_score_list = []
        # df_cluster shows which ids are inside the cluster
        for idx in tqdm(range(df_summary.shape[0])):
            df_line = df_summary.iloc[idx]
            norm_confidence_scores = []
            norm_total = 0
            for norm_value in df_line["raw_sample_norms"]:
                norm_value = parse_to_int(norm_value)
                if norm_value is not None:
                    norm_confidence_scores.append(norm_value)
                    norm_total += 1
            norm_confidence_score = np.mean(norm_confidence_scores)
            norm_confidence_score_list.append(norm_confidence_score)
            norm_total_list.append(norm_total)
        df_summary["cluster_size"] = norm_total_list
        df_summary["norm"] = norm_confidence_score_list

        self.save_output(df_summary)

    def save_output(self, df):
        logger.info(f"save to {self._local_config['output_file']}")
        df.to_csv(
            self._local_config["output_file"],
            index=False,
        )
