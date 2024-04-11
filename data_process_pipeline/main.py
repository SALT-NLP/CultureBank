"""
python main.py -i 0 -c /sailhome/weiyans/CultureBank/data_process/config_tiktok.yaml --dry_run 200
"""

import logging

from utils.config_reader import read_config
import argparse
from pipeline.main_pipeline import CultureBankPipeline


logging.basicConfig(
    level=logging.INFO,
    format="[%(processName)s] [%(asctime)s] [%(name)s] " "[%(levelname)s] %(message)s",
    datefmt="%d-%m %H:%M:%S",
)

logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="The CultureBank extraction pipeline")

    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config file", required=True
    )

    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        help="The GPUs to use, comma-separated",
        required=False,
    )

    parser.add_argument(
        "-i",
        "--components",
        type=str,
        help="The components to run",
        # nargs="+"
    )

    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output file", required=False
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        type=int,
        default=None,
        help="the number of rows to run in the dry run",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--sent_threshold",
        type=float,
        default=None,
        help="the threshold for sentence clustering",
        required=False,
    )
    parser.add_argument(
        "-ct",
        "--cultural_group_threshold",
        type=float,
        default=None,
        help="the threshold for cultural group clustering",
        required=False,
    )
    parser.add_argument(
        "-tt",
        "--topic_threshold",
        type=float,
        default=None,
        help="the threshold for topic clustering",
        required=False,
    )
    parser.add_argument(
        "-w",
        "--with_other_desc",
        action="store_true",
        help="in the clustering, do we consider other desc or not",
    )
    args = parser.parse_args()

    logger.info(f"Reading config from {args.config}...")
    config = read_config(args.config)

    # Overwrite the config
    logger.info(f"Overwriting config with arguments...")
    if args.gpus:
        config["gpus"] = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if args.components:
        config["chosen_components"] = sorted(
            [int(c) for c in args.components.split(",")]
        )
    if args.output_file:
        config["output_file"] = args.output_file
    if args.sent_threshold:
        config["sent_threshold"] = args.sent_threshold
    if args.cultural_group_threshold:
        config["cultural_group_threshold"] = args.cultural_group_threshold
    if args.with_other_desc:
        config["with_other_desc"] = args.with_other_desc
    else:
        config["with_other_desc"] = False
    config["dry_run"] = args.dry_run

    logger.info("Initializing the pipeline...")
    pipeline = CultureBankPipeline(config)

    pipeline.run()


if __name__ == "__main__":
    main()
