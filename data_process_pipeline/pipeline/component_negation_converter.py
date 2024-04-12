"""
cd data_process
python pipeline/component_negation_converter.py
"""

import spacy
import pandas as pd

from spacy import displacy
from pathlib import Path
from tqdm import tqdm

from pipeline.pipeline_component import PipelineComponent

import logging

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)


def convert_to_affirmative(sentence, norm):
    doc = nlp(sentence)
    new_sentence = []

    for i, token in enumerate(doc):
        # Check for negation words and modify the sentence accordingly
        if token.dep_ == "neg":
            # Skip adding the negation word and flip the norm
            norm = 1 if norm == 0 else 0
            continue
        if (
            token.dep_ in ["aux", "auxpass"]
            and ((i + 1 < len(doc)) and token.nbor(1).dep_ == "neg")
            and token.text.lower() in ["do", "did", "does"]
        ):
            # Skip "do", "did", "does" next to negation
            continue
        if (i + 1 < len(doc)) and token.nbor(1).dep_ == "neg":
            verb_map = {
                "ca": "can ",
                "wo": "will ",
            }
            if token.nbor(1).text == "n't":
                if token.text in verb_map:
                    new_sentence.append(verb_map[token.text])
                else:
                    new_sentence.append(token.text_with_ws + " ")
            else:
                new_sentence.append(token.text_with_ws)
        else:
            new_sentence.append(token.text_with_ws)

    # Reconstruct the sentence
    return "".join(new_sentence), norm


class NegationConverter(PipelineComponent):
    description = (
        "Convert the extracted data into positive form, if it contains negation"
    )
    config_layer = "negation_converter"

    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]
        if "output_file" in self._local_config:
            self.check_if_output_exists(self._local_config["output_file"])

    def read_input(self):
        df = pd.read_csv(self._local_config["input_file"])
        if self._config["dry_run"] is not None:
            df = df.head(self._config["dry_run"])
        return df

    def run(self):
        df = self.read_input()
        total = 0
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if pd.isna(df.iloc[idx]["actor's behavior"]) or pd.isna(
                df.iloc[idx]["norm"]
            ):
                continue
            sentence = df.iloc[idx]["actor's behavior"]
            try:
                norm = df.iloc[idx]["norm"]
                norm = int(float(norm))
                prev_norm = norm
            except:
                continue
            sentence, norm = convert_to_affirmative(str(sentence), int(float(norm)))
            if prev_norm != norm:
                total += 1
            df.iloc[idx]["actor's behavior"] = sentence
            df.iloc[idx]["norm"] = norm
        logger.info(f"{total} are negation! {total/df.shape[0]:.3f}")
        self.save_output(df)

    def save_output(self, df):
        logger.info(f"save to {self._local_config['output_file']}")
        df.to_csv(
            self._local_config["output_file"],
            index=False,
        )


if __name__ == "__main__":
    examples = [
        "They never eat pasta.",
        "i will not do this",
        "i couldn't do this",
        "i won't do this",
        "i haven't do this",
        "i hasn't do this",
        "i wouldn't do this",
        "i shouldn't do this",
        "I don't like apples.",
        "i can't do this",
        "give what they can",
        "She is not going to the party.",
        "I don't not like it.",
        "don't use bidets",
        "i like apples",
        "i do like apples",
        "do not slaughter animals for sacrifice",
    ]

    for sent in examples:
        print(sent, 0)
        sent, norm = convert_to_affirmative(sent, 0)
        print(sent, norm)
        print()
