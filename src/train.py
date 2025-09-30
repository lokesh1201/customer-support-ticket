import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from .data import load_dataset, build_text_field, train_test_split
from .model import build_pipeline, get_default_params


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train ticket classifier")
	parser.add_argument("--data_path", type=str, required=True)
	parser.add_argument("--model_path", type=str, required=True)
	parser.add_argument("--test_size", type=float, default=0.2)
	parser.add_argument("--random_state", type=int, default=42)
	parser.add_argument("--params", type=str, default="", help="JSON string of pipeline params")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_dataset(args.data_path)
	train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)

	X_train = build_text_field(train_df)
	y_train = train_df["Category"].astype(str)
	X_test = build_text_field(test_df)
	y_test = test_df["Category"].astype(str)

	pipeline = build_pipeline()
	params = get_default_params()
	if args.params:
		params.update(json.loads(args.params))
	pipeline.set_params(**params)

	pipeline.fit(X_train, y_train)
	preds = pipeline.predict(X_test)
	report = classification_report(y_test, preds, output_dict=True)

	model_path = Path(args.model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump({"pipeline": pipeline, "labels": sorted(df["Category"].astype(str).unique())}, model_path)

	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()

