"""
Final touch
"""

import spacy
import pandas as pd

# from profanity_check import predict, predict_prob
# from better_profanity import profanity
from utils.constants import CULTUREBANK_FIELDS
from datetime import datetime

from tqdm import tqdm

from pipeline.pipeline_component import PipelineComponent

import logging
import math

logger = logging.getLogger(__name__)


# add time
def count_to_bin(count):
    # Calculate the lower bound of the bin (rounded down to the nearest multiple of 10)
    lower_bound = (count // 10) * 10
    # Calculate the upper bound of the bin
    upper_bound = lower_bound + 10
    # Return the bin as a string
    return f"[{lower_bound}, {upper_bound})"


class FinalFormatter(PipelineComponent):
    description = "Final data prepration"
    config_layer = "final_formatter"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self.config = config
        self._local_config = config[self.config_layer]
        self._min_cluster_size = config["clustering_component"]["min_cluster_size"]
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df

    def format_time(self, df):
        time_list = []
        for i in tqdm(range(df.shape[0])):
            df_line = df.iloc[i]
            year_to_count = {}
            for timestamp in eval(df_line["raw_sample_times"]):
                year = datetime.fromtimestamp(timestamp).year
                if year in year_to_count:
                    year_to_count[year] += 1
                else:
                    year_to_count[year] = 1
            year_to_bin = {
                year: count_to_bin(count) for year, count in year_to_count.items()
            }
            time_list.append({k: year_to_bin[k] for k in sorted(year_to_bin)})
        return time_list

    def run(self):
        df = self.read_input()
        # final clean
        df = df[df.cluster_size >= self._min_cluster_size]
        # round the agreement number
        df["agreement"] = df["norm"].round(1)

        # bin the cluster size
        bins = [5] + list(
            range(20, math.ceil(df["cluster_size"].max() / 10) * 10 + 1, 10)
        )
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
        df["num_support_bin"] = pd.cut(
            df["cluster_size"], bins=bins, labels=labels, right=False
        )

        # bin the time_range
        df["time_range"] = self.format_time(df)

        df_final = df[
            [
                "cluster_id",
                # "representative_cultural group",
                "context",
                "goal",
                "relation",
                "actor",
                "actor's behavior",
                "recipient",
                "recipient's behavior",
                "other descriptions",
                # "representative_topic",
                "agreement",
                "num_support_bin",
                "time_range",
            ]
        ]
        df_final.columns = [
            "cluster_id",
            # "cultural group",
            "context",
            "goal",
            "relation",
            "actor",
            "actor_behavior",
            "recipient",
            "recipient_behavior",
            "other_descriptions",
            # "topic",
            "agreement",
            "num_support_bin",
            "time_range",
        ]

        # filter controversial data
        if self._local_config["controversial_annotation_file"] is not None:
            df_controversial_annotated = pd.read_csv(
                self._local_config["controversial_annotation_file"]
            )
            df_final = df_final[
                ~df_final.cluster_id.isin(
                    df_controversial_annotated[
                        df_controversial_annotated[
                            self.config["content_moderation"][
                                "controversial_field_name_to_annotate"
                            ]
                        ]
                        == 1
                    ].cluster_id
                )
            ]

        self.save_output(
            df_final,
            save_dir=self._local_config["output_file"],
        )

    def save_output(self, df, save_dir):
        logger.info(f"save to {save_dir}")
        df.to_csv(
            save_dir,
            index=False,
        )
