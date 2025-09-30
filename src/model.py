from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(max_features: int = 50000, ngram_range=(1, 2)) -> Pipeline:
	"""Create a text classification pipeline with TF-IDF and LinearSVC."""
	return Pipeline(
		steps=[
			(
				"tfidf",
				TfidfVectorizer(
					lowercase=True,
					stop_words="english",
					ngram_range=ngram_range,
					max_features=max_features,
					norm="l2",
				),
			),
			("clf", LinearSVC()),
		]
	)


def get_default_params() -> Dict[str, Any]:
	return {
		"tfidf__max_features": 50000,
		"tfidf__ngram_range": (1, 2),
		"clf__C": 1.0,
	}

