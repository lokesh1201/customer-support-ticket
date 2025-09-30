import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from .data import load_dataset, build_text_field


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate ticket classifier")
	parser.add_argument("--data_path", type=str, required=True)
	parser.add_argument("--model_path", type=str, required=True)
	parser.add_argument("--report_path", type=str, required=False, default="")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_dataset(args.data_path)
	X = build_text_field(df)
	y_true = df["Category"].astype(str)

	obj = joblib.load(args.model_path)
	pipeline = obj["pipeline"]
	y_pred = pipeline.predict(X)

	acc = accuracy_score(y_true, y_pred)
	precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=sorted(y_true.unique()))
	report = classification_report(y_true, y_pred, output_dict=True)

	metrics = {
		"accuracy": acc,
		"per_class": {
			"labels": list(sorted(y_true.unique())),
			"precision": precision.tolist(),
			"recall": recall.tolist(),
			"f1": f1.tolist(),
			"support": support.tolist(),
		},
		"classification_report": report,
	}

	if args.report_path:
		path = Path(args.report_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(metrics, indent=2))

	print(json.dumps({"accuracy": acc}, indent=2))


if __name__ == "__main__":
	main()

