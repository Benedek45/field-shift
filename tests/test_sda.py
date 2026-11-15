from unittest.mock import Mock

import pytest

from fieldshift.data import sda


@pytest.fixture(autouse=True)
def no_install(monkeypatch):
    monkeypatch.setattr(sda, "install_dns_fallback", lambda: None)


def test_column_names(monkeypatch):
    post = Mock(return_value=Mock(
        status_code=200, json=lambda: {"Table": [["mukey", "name"], ["123", "loam"]]}
    ))
    monkeypatch.setattr(sda.requests, "post", post)
    result = sda.query("SELECT TOP 1 mukey, name FROM mapunit")
    assert result.to_dict("records") == [{"mukey": "123", "name": "loam"}]
    assert post.call_args.kwargs["json"]["format"] == "JSON+COLUMNNAME"


@pytest.mark.parametrize("status, body, message", [
    (503, {}, "HTTP 503"),
    (200, {"Error": "bad SQL"}, "API error"),
    (200, {"unexpected": "response"}, "missing column header"),
    (200, {"Table": [["mukey"]]}, None),
])
def test_errors_and_empty_result(monkeypatch, status, body, message):
    monkeypatch.setattr(sda.requests, "post", lambda *a, **k: Mock(
        status_code=status, text="unavailable", json=lambda: body,
    ))
    if message:
        with pytest.raises(RuntimeError, match=message):
            sda.query("sql")
    else:
        assert list(sda.query("sql").columns) == ["mukey"]


def test_sda_empty_object_is_no_rows(monkeypatch):
    monkeypatch.setattr(sda.requests, "post", lambda *a, **k: Mock(
        status_code=200, json=dict,
    ))
    assert sda.query("SELECT mukey FROM mapunit WHERE 1=0").empty
