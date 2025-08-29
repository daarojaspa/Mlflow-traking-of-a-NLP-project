import pytest
import pandas as pd
from tickettopicmodeling import TicketTopicModeling  # adjust import to your file


@pytest.fixture
def sample_df():
    """Fixture that returns a small raw dataframe for testing."""
    data = [
        {
            "_source.complaint_what_happened": "I was charged twice",
            "_source.product": "Credit card",
            "_source.sub_product": "Rewards",
        },
        {
            "_source.complaint_what_happened": "   The ATM ate my card ",
            "_source.product": "Bank account",
            "_source.sub_product": "ATM",
        },
        {
            "_source.complaint_what_happened": "",
            "_source.product": "Loan",
            "_source.sub_product": "Mortgage",
        },
    ]
    return pd.DataFrame(data)


def test_data_transform_removes_empty_and_formats(sample_df):
    model = TicketTopicModeling()

    transformed = model.data_transform(sample_df)

    # ✅ Should remove the empty complaint
    assert transformed.shape[0] == 2  

    # ✅ Columns should be renamed
    assert "complaint_what_happened" in transformed.columns
    assert "ticket_classification" in transformed.columns

    # ✅ Should lowercase and strip
    assert transformed.loc[1, "complaint_what_happened"] == "the atm ate my card"

    # ✅ ticket_classification is product + sub_product
    assert " + " in transformed.loc[0, "ticket_classification"]


def test_split_creates_train_and_test(tmp_path, sample_df):
    model = TicketTopicModeling(output_dir=str(tmp_path))

    train_df, test_df = model.split(sample_df, test_size=0.5, random_state=1)

    # ✅ Check split sizes
    assert len(train_df) + len(test_df) == len(sample_df)

    # ✅ Files should exist
    latest_dir = tmp_path / "splitted" / "latest"
    assert (latest_dir / "train.csv").exists()
    assert (latest_dir / "test.csv").exists()
