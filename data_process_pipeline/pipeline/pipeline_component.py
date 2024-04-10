import os
import logging
import sys


class PipelineComponent:
    description = "The interface for a pipeline component"

    def __init__(self, config: dict):
        self._config = config

    def run(self):
        """Runs the component"""
        raise NotImplementedError()

    def get_description(self):
        return self.description

    def check_if_output_exists(self, path):
        if os.path.exists(path):
            logging.warning(
                f"output path {path} already exists! are you sure you want to overwrite?"
            )
            user_input = input("type Y/YES to overwrite\n")
            if not user_input in ["Y", "YES"]:
                print("Exiting gracefully")
                sys.exit(0)

    def read_input(self):
        """Read the input"""
        raise NotImplementedError()

    def save_output(self):
        """Save the output"""
        raise NotImplementedError()

    def __str__(self):
        return f"<{self.__class__.__name__}> - {self.description}"

    def __repr__(self):
        return self.__str__()

    def initialize(self):
        raise NotImplementedError()

    def is_initialized(self) -> bool:
        """Returns True if this component is initialized"""
        raise NotImplementedError()
