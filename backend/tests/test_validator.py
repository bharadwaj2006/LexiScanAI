import pytest
from validator.rules import validate_and_normalize

def test_validate_date():
    raw_data = {
        "dates": [{"text": "January 15, 2024", "label": "DATE"}],
        "amounts": [],
        "parties": [],
        "termination_clauses": []
    }
    result = validate_and_normalize(raw_data)
    assert len(result["dates"]) == 1
    assert result["dates"][0]["normalized"] == "2024-01-15"

def test_validate_amount():
    raw_data = {
        "dates": [],
        "amounts": [{"text": "$5,000,000", "label": "AMOUNT"}],
        "parties": [],
        "termination_clauses": []
    }
    result = validate_and_normalize(raw_data)
    assert len(result["amounts"]) == 1
    assert result["amounts"][0]["currency"] == "USD"
    assert result["amounts"][0]["normalized_value"] == 5000000.0

def test_validate_party():
    raw_data = {
        "dates": [],
        "amounts": [],
        "parties": [{"text": "Acme Corporation", "label": "PARTY"}],
        "termination_clauses": []
    }
    result = validate_and_normalize(raw_data)
    assert len(result["parties"]) == 1
    assert result["parties"][0]["valid"] == True
