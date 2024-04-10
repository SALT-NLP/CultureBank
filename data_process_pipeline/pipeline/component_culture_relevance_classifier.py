import os
import time
import pandas as pd
from transformers import pipeline
import logging

from pipeline.pipeline_component import PipelineComponent

def get_best_ckpt(model_output_dir):
    ckpt_dirs = os.listdir(model_output_dir)
    ckpt_dirs = [dir for dir in ckpt_dirs if "checkpoint" in dir]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpt_dirs[-1]

    return os.path.join(model_output_dir, last_ckpt)

logger = logging.getLogger(__name__)

class CultureRelevanceClassifier(PipelineComponent):
    description = "Classify sentences/comments and pick out the culturally-relevant ones"
    config_layer = "culture_classifier"

    def __init__(self, config: dict):
        super().__init__(config)

        # Get local config
        self._local_config = config[self.config_layer]

        # Get the labels
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])

        # Get the classifier config
        self._model_name = self._local_config["model_name"]
        self._device = self._local_config["device"]
        self._last_ckpt = get_best_ckpt(self._local_config["classifier_path"])

        # prepare the classifier
        self.classifier = pipeline(
            "text-classification",
            model=self._last_ckpt,
            device=self._device,
        )

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df
    
    def run(self):
        id2label = {0: "No", 1: "Yes"}
        label2id = {"No": 0, "Yes": 1}
        
        df = self.read_input()
        logger.info(f"total number of samples: {len(df)}")
        test_pred = self.classifier(
            [str(r) for r in df[self._local_config["field_name_with_comments"]].tolist()],
            batch_size=self._local_config["batch_size"],
            truncation=True,
            max_length=self._local_config["batch_size"],
        )
        df["pred_label"] = [id2label[int(pred["label"][-1])] for pred in test_pred]
        df["pred_score"] = [round(pred["score"], 2) for pred in test_pred]
        self.save_output(df)
        logger.info("Prediction Done!")

    def save_output(self, df):
        logger.info(f"save to {self._local_config['output_file']}")
        df.to_csv(
            self._local_config["output_file"],
            index=False,
        )

