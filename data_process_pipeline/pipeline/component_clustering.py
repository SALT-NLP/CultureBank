"""
clustering component
"""

import pandas as pd
import torch
import json

from tqdm import tqdm
import logging
from sklearn.metrics import classification_report

from pipeline.pipeline_component import PipelineComponent
from sentence_transformers import SentenceTransformer

from utils.clustering import (
    hac_clustering_retain_index,
    secondary_clustering,
)

logger = logging.getLogger(__name__)


class ClusteringComponent(PipelineComponent):
    description = "clustering the extracted and processed knowledge"
    config_layer = "clustering_component"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]
        self._override_config()

        # log the parameters
        self._condition = f"group={self._local_config['cultural_group_threshold']}_sent={self._local_config['sent_threshold']}_otherdesc={self._local_config['with_other_desc']}"
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
        for key in ["cultural_group_threshold", "sent_threshold", "with_other_desc"]:
            if key in self._config and self._config[key] is not None:
                self._local_config[key] = self._config[key]

    def _create_new_output_dir(self):
        import pathlib

        new_output_dir = "/".join(
            self._local_config["output_file"].split("/")[:-1] + [self._condition]
        )
        logger.info(f"making new dirs {new_output_dir}")
        pathlib.Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        for key in ["output_file", "output_score_file", "output_filtered_file"]:
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
        raw_clusters = self.cluster_groups(df)
        clustered_df_filtered, clustered_df_unfiltered = self.cluster_norms(
            df, raw_clusters
        )

        if self._local_config["annotated_file"] != "none":
            self.evaluate_cluster_with_annotation(clustered_df_unfiltered)
        self.save_output(clustered_df_filtered, clustered_df_unfiltered)
        logger.info("Clustering Done!")

    def save_output(self, df_filtered, df_unfiltered):
        logger.info(f"save to {self._local_config['output_file']}")
        df_filtered.to_csv(
            self._local_config["output_filtered_file"],
            index=False,
        )
        df_unfiltered.to_csv(
            self._local_config["output_file"],
            index=False,
        )
        with open(self._local_config["output_score_file"], "w") as fh:
            json.dump(self.scores, fh)

    def cluster_groups(self, df):
        # cluster cultural group first
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

    def cluster_norms(self, df, raw_clusters):
        fields = [
            "context",
            "goal",
            "relation",
            "actor",
            "recipient",
            "actor's behavior",
            "recipient's behavior",
        ]
        if self._local_config["with_other_desc"]:
            fields.append("other descriptions")
        all_clusters = []

        # enumerate each cultural group, and cluster within each cultural group, to avoid OOM
        for i, cluster in enumerate(tqdm(raw_clusters)):
            sents_with_index = []
            for idx, _ in cluster:
                row = df.iloc[idx]
                sents_with_index.append(
                    (
                        idx,
                        ", ".join(
                            f"{df.iloc[idx][field]}"
                            for field in fields
                            if pd.notna(df.iloc[idx][field])
                        ),
                    )
                )

            embeddings = self.sbert.encode(
                [sent[1] for sent in sents_with_index], show_progress_bar=True
            )
            secondary_clusters, score = secondary_clustering(
                sents_with_index, embeddings, self._local_config["sent_threshold"]
            )

            logger.info(f"the silhouette_score for the clustering is {score}")
            all_clusters.extend(secondary_clusters)
            self.scores["cluster_silhouette_score"].append(score)

        # build a new dataframe for clusters
        clustered_df_unfiltered = pd.DataFrame(
            columns=list(df.columns)
            + ["cluster_size", "raw_sample_vids", "raw_samples"]
        )

        for cluster in tqdm(all_clusters):
            rep_idx = cluster[0][0]
            rep_sent = cluster[0][1]
            rep_row = df.iloc[rep_idx]
            raw_samples = []
            raw_sample_ids = []
            raw_sample_times = []
            raw_sample_norms = []
            for idx, _ in cluster:
                item = df.iloc[idx]
                sample_str = item.to_dict()
                raw_samples.append(sample_str)
                raw_sample_ids.append(item["vid_unique"])
                raw_sample_times.append(item["comment_utc"])
                raw_sample_norms.append(item["norm"])

            # Create a new row with the representative's data and the raw samples
            new_row = rep_row.to_dict()
            new_row["raw_samples"] = raw_samples
            new_row["raw_sample_vids"] = raw_sample_ids
            new_row["raw_sample_times"] = raw_sample_times
            new_row["raw_sample_norms"] = raw_sample_norms
            new_row["cluster_size"] = len(raw_samples)
            clustered_df_unfiltered = pd.concat(
                [clustered_df_unfiltered, pd.DataFrame([new_row])], ignore_index=True
            )
        # Filter out small clusters (which are likely noise)

        clustered_df_unfiltered["cluster_id"] = clustered_df_unfiltered.index
        clustered_df_filtered = clustered_df_unfiltered[
            clustered_df_unfiltered["cluster_size"]
            >= self._local_config["min_cluster_size"]
        ]
        print(f"num clusters before filtering: {len(clustered_df_unfiltered)}")
        print(f"num clusters after filtering: {len(clustered_df_filtered)}")
        return clustered_df_filtered, clustered_df_unfiltered

    def evaluate_cluster_with_annotation(self, clustered_df_unfiltered):
        df_annotation = pd.read_csv(self._local_config["annotated_file"])
        vid_unique_annotation = df_annotation[
            ~df_annotation.vid_unique.isnull()
        ].vid_unique.tolist()
        cluster_annotation = df_annotation[
            ~df_annotation.vid_unique.isnull()
        ].cluster_annot.tolist()
        assert len(vid_unique_annotation) == len(cluster_annotation)
        vid_annotate_cluster_map = {
            vid_unique: int(cluster_id)
            for vid_unique, cluster_id in zip(vid_unique_annotation, cluster_annotation)
        }
        vid_original_cluster_map = {}
        for vid_unique in vid_annotate_cluster_map:
            original_cluster_id = clustered_df_unfiltered[
                clustered_df_unfiltered.raw_sample_vids.astype(str).str.contains(
                    vid_unique
                )
            ].cluster_id.values.item()
            vid_original_cluster_map[vid_unique] = original_cluster_id
        old_to_annotate_cluster_id_map = {}
        for vid_unique in vid_annotate_cluster_map:
            annotated_cluster_id = vid_annotate_cluster_map[vid_unique]
            original_cluster_id = vid_original_cluster_map[vid_unique]
            if (original_cluster_id not in old_to_annotate_cluster_id_map) and (
                annotated_cluster_id not in old_to_annotate_cluster_id_map.values()
            ):
                # original cluster is new in the map, and the annotated cluster is also new in the map
                old_to_annotate_cluster_id_map[original_cluster_id] = (
                    annotated_cluster_id
                )
            elif annotated_cluster_id in old_to_annotate_cluster_id_map.values():
                # the annotated cluster already appears in the map
                for key in old_to_annotate_cluster_id_map:
                    if old_to_annotate_cluster_id_map[key] == annotated_cluster_id:
                        mapped_cluster_id = key
                if original_cluster_id != mapped_cluster_id:
                    old_to_annotate_cluster_id_map[original_cluster_id] = (
                        original_cluster_id
                    )
        y_true = []
        y_pred = []
        for vid_unique in vid_annotate_cluster_map:
            y_true.append(vid_annotate_cluster_map[vid_unique])
            y_pred.append(
                old_to_annotate_cluster_id_map[vid_original_cluster_map[vid_unique]]
            )
        text = classification_report(y_true, y_pred)
        self.scores["clf_report"] = text
        logger.info(text)
