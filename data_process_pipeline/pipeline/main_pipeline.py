import logging
import os

from pipeline.component_negation_converter import NegationConverter

from pipeline.component_clustering import ClusteringComponent
from pipeline.pipeline_component import PipelineComponent
from pipeline.component_knowledge_extractor import KnowledgeExtractor
from pipeline.component_cluster_summarizer import ClusterSummarizer
# from pipeline.component_confidence_calculator import ConfidenceCalculator
from pipeline.component_agreement_calculator import AgreementCalculator

# from pipeline.component_confidence_calculator_for_reddits import (
#     ConfidenceCalculatorForReddits,
# )
from pipeline.component_content_moderation import ContentModeration

from pipeline.component_final_formatter import FinalFormatter
from pipeline.component_culture_relevance_classifier import CultureRelevanceClassifier

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: dict):
        self._config = config

        running_component_indexes = self._config["chosen_components"]

        self._running_components = [
            self._get_possible_component_at(index)
            for index in running_component_indexes
        ]

        self.print_possible_components()
        print("Run the following component:")
        print(self.get_running_components())

    def run(self):
        logger.info(
            f"Running pipeline {self.__class__.__name__} with "
            f"{len(self._running_components)} "
            f"components..."
        )
        for component in self._running_components:
            # if not component.is_initialized():
            #     component.initialize()
            component.run()
        logger.info(f"Pipeline {self.__class__.__name__} finished.")

    @classmethod
    def get_possible_components(cls):
        raise NotImplementedError

    def _get_possible_component_at(self, index) -> PipelineComponent:
        try:
            return self.get_possible_components()[index](self._config)
        except IndexError:
            raise ValueError(
                f"The chosen component index {index} is out of range. "
                f"Possible components: "
                + str(
                    [
                        ((i), c.__name__)
                        for i, c in enumerate(self.get_possible_components())
                    ]
                )
            )

    def get_running_components(self):
        return self._running_components

    @classmethod
    def print_possible_components(cls):
        print(f"Possible pipeline components of {cls.__name__}:")
        for i, component in enumerate(cls.get_possible_components()):
            print(f"  ({i + 1}) {component.__name__} - {component.description}")


class CultureBankPipeline(Pipeline):
    def __init__(self, config: dict):
        logger.info("Initializing CultureBankPipeline...")
        os.makedirs(config["result_base_dir"], exist_ok=True)
        self._init_component_dir(config)
        super().__init__(config)

    def _init_component_dir(self, config):
        for component in self.get_possible_components():
            os.makedirs(
                os.path.join(config["result_base_dir"], component.config_layer),
                exist_ok=True,
            )

    @classmethod
    def get_possible_components(cls):
        return [
            CultureRelevanceClassifier,
            KnowledgeExtractor,
            NegationConverter,
            # NormEntailment,
            ClusteringComponent,
            ClusterSummarizer,
            # ConfidenceCalculatorForReddits,
            # ContentFilter,
            # ControversialFilter,
            # TopicClustering,
        ]


#
