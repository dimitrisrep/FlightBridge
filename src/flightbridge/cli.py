"""Reproducible command-line entry points."""

import argparse
import json
from pathlib import Path

from flightbridge.service import FlightBridge
from flightbridge.text_model import train_intents
from flightbridge.training import train


def main():
    parser = argparse.ArgumentParser(prog="flightbridge")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("schedule_change_dataset.csv"))
    training.add_argument("--output", type=Path, default=Path("artifacts"))
    text = sub.add_parser("train-intents")
    text.add_argument("--training", type=Path, default=Path("data/intents/train.jsonl"))
    text.add_argument("--evaluation", type=Path, default=Path("data/intents/test.jsonl"))
    text.add_argument("--output", type=Path, default=Path("artifacts"))
    analyze = sub.add_parser("analyze")
    analyze.add_argument("message")
    analyze.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "train":
        report = train(args.data, args.output)
        print(json.dumps({key: info["holdout"] for key, info in report["targets"].items()}, indent=2))
    elif args.command == "train-intents":
        print(json.dumps(train_intents(args.training, args.evaluation, args.output), indent=2))
    else:
        service = (
            FlightBridge.load(args.artifacts)
            if (args.artifacts / "entitlements.joblib").exists()
            else FlightBridge()
        )
        print(json.dumps(service.analyze(args.message), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
