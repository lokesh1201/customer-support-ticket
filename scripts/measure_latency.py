import argparse
import statistics
import time

from src.inference import TicketClassifier


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Measure prediction latency")
	parser.add_argument("--model_path", type=str, required=True)
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--description", type=str, required=True)
	parser.add_argument("--num_runs", type=int, default=200)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	clf = TicketClassifier(args.model_path)

	# warm-up
	for _ in range(10):
		clf.predict(args.subject, args.description)

	durations_ms = []
	start_all = time.perf_counter()
	for _ in range(args.num_runs):
		start = time.perf_counter()
		_ = clf.predict(args.subject, args.description)
		durations_ms.append((time.perf_counter() - start) * 1000.0)
	wall_ms = (time.perf_counter() - start_all) * 1000.0

	print(
		{
			"runs": args.num_runs,
			"p50_ms": round(statistics.median(durations_ms), 3),
			"p90_ms": round(sorted(durations_ms)[int(0.9 * len(durations_ms))], 3),
			"mean_ms": round(statistics.mean(durations_ms), 3),
			"min_ms": round(min(durations_ms), 3),
			"max_ms": round(max(durations_ms), 3),
			"wall_ms": round(wall_ms, 3),
		}
	)


if __name__ == "__main__":
	main()

