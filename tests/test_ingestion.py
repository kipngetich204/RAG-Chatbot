from app.ingestion.loader import DocumentLoader


def test_loader():
    docs = DocumentLoader.load_directory(
        "data/raw"
    )

    assert len(docs) > 0