from src.model import build_pipeline


def test_pipeline_trains_and_predicts():
	texts = [
		"App crash on login",
		"Export CSV please",
		"VPN fails to connect",
		"Charged twice",
		"Password reset issue",
	]
	labels = [
		"Bug Report",
		"Feature Request",
		"Technical Issue",
		"Billing Inquiry",
		"Account Management",
	]

	p = build_pipeline()
	p.fit(texts, labels)
	pred = p.predict(["Billing discrepancy on invoice"])[0]
	assert isinstance(pred, str)

