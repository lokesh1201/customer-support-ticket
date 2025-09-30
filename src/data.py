import pandas as pd
from typing import Tuple


REQUIRED_COLUMNS = [
	"Ticket_ID",
	"Subject",
	"Description",
	"Category",
	"Priority",
	"Timestamp",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
	"""Load dataset CSV and validate required columns.

	Args:
		csv_path: Path to the CSV file.

	Returns:
		Validated DataFrame.
	"""
	df = pd.read_csv(csv_path)
	missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	return df


def build_text_field(df: pd.DataFrame) -> pd.Series:
	"""Concatenate subject and description to a single text field."""
	subject = df["Subject"].fillna("").astype(str)
	description = df["Description"].fillna("").astype(str)
	return (subject + " " + description).str.strip()


def train_test_split(
	df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Simple stratified split by Category using pandas sample.

	This avoids importing sklearn here and keeps data ops lightweight.
	"""
	# Stratified split by Category
	train_parts = []
	test_parts = []
	for category, group in df.groupby("Category"):
		g = group.sample(frac=1.0, random_state=random_state)
		# Ensure at least one sample per class stays in train when possible
		if len(g) <= 1:
			train_parts.append(g)
			continue
		n_test = max(1, int(len(g) * test_size))
		n_test = min(n_test, len(g) - 1)
		test_parts.append(g.iloc[:n_test])
		train_parts.append(g.iloc[n_test:])
	train_df = pd.concat(train_parts, ignore_index=True)
	# If no test parts (e.g., all classes have a single sample), fallback to random split
	if len(test_parts) == 0:
		g = df.sample(frac=1.0, random_state=random_state)
		n_test_global = max(1, int(len(g) * test_size))
		n_test_global = min(n_test_global, max(1, len(g) - 1))
		test_df = g.iloc[:n_test_global]
		train_df = g.iloc[n_test_global:]
		return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

	test_df = pd.concat(test_parts, ignore_index=True)
	return train_df, test_df

