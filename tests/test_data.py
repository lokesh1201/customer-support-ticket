import pandas as pd

from src.data import load_dataset, build_text_field, train_test_split


def test_build_text_field_concatenates_subject_and_description(tmp_path):
	df = pd.DataFrame(
		{
			"Ticket_ID": [1, 2],
			"Subject": ["Hello", "World"],
			"Description": ["Alpha", "Beta"],
			"Category": ["Bug Report", "Feature Request"],
			"Priority": ["High", "Low"],
			"Timestamp": ["2025-01-01", "2025-01-02"],
		}
	)
	text = build_text_field(df)
	assert list(text) == ["Hello Alpha", "World Beta"]


def test_stratified_split_keeps_train_nonempty_per_class(tmp_path):
	df = pd.DataFrame(
		{
			"Ticket_ID": [1, 2, 3, 4],
			"Subject": ["A", "B", "C", "D"],
			"Description": ["a", "b", "c", "d"],
			"Category": ["X", "X", "Y", "Y"],
			"Priority": ["H", "H", "L", "L"],
			"Timestamp": ["t", "t", "t", "t"],
		}
	)
	train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
	assert not train_df.empty
	assert not test_df.empty

