"""
Main entry point for the HOI-GNN pipeline.
Provides a CLI to run dataset loading, training (with resumption), evaluation, or inference.
"""

import argparse
import logging
import sys
import yaml

from datasets.dataset_loader import fetch_dataset
from train import main as train_main
from evaluate import main as evaluate_main
from inference import main as inference_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="HOI-GNN Pipeline CLI - Run dataset loading, training, evaluation, or inference."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["load_data", "train", "evaluate", "inference"],
        help="Mode to run: load_data, train, evaluate, or inference."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ava",
        help="Dataset to download (only used in load_data mode). Options: ava, hico-det, v-coco."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (used in training, evaluation, and inference)."
    )
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "load_data":
        try:
            logging.info(f"Starting dataset download for {args.dataset.upper()}...")
            fetch_dataset(args.dataset)
            logging.info("Dataset download and extraction completed.")
        except Exception as e:
            logging.error(f"Error fetching dataset: {e}")
            sys.exit(1)
    elif args.mode == "train":
        logging.info("Starting training...")
        train_main()
        logging.info("Training completed.")
    elif args.mode == "evaluate":
        logging.info("Starting evaluation...")
        evaluate_main()
        logging.info("Evaluation completed.")
    elif args.mode == "inference":
        logging.info("Starting real-time inference...")
        inference_main()
        logging.info("Inference terminated.")
    else:
        logging.error("Invalid mode provided. Use --help for usage details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
