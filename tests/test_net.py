import logging
import socket

import pytest
import requests

from fieldshift import net


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.setattr(net, "_cache", {})
    monkeypatch.setattr(net, "_warned", set())


def test_system_success_untouched(monkeypatch):
    result = [object()]
    calls = []

    def original(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(net, "_original_getaddrinfo", original)
    monkeypatch.setattr(net, "_doh_addresses", lambda *_: pytest.fail("DoH used"))
    assert net._getaddrinfo("healthy.example", 443) is result
    assert calls == [("healthy.example", 443, 0, 0, 0, 0)]


def test_fallback_and_warning_once(monkeypatch, caplog):
    error = socket.gaierror(socket.EAI_NONAME, "failed")
    result = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))]

    def original(host, *args):
        if host == "broken.example":
            raise error
        assert host == "192.0.2.1"
        return result

    monkeypatch.setattr(net, "_original_getaddrinfo", original)
    monkeypatch.setattr(net, "_doh_addresses", lambda *_: ["192.0.2.1"])
    with caplog.at_level(logging.WARNING):
        assert net._getaddrinfo("broken.example", 443) == result
        assert net._getaddrinfo("broken.example", 443) == result
    assert len(caplog.records) == 1


def test_both_fail_preserves_original_error(monkeypatch):
    error = socket.gaierror(socket.EAI_NONAME, "system error")

    def fail(*args):
        raise error

    def doh_fail(*args):
        raise OSError("providers failed")

    monkeypatch.setattr(net, "_original_getaddrinfo", fail)
    monkeypatch.setattr(net, "_doh_addresses", doh_fail)
    with pytest.raises(socket.gaierror) as caught:
        net._getaddrinfo("broken.example", 443)
    assert caught.value is error


@pytest.mark.parametrize("host", ["1.1.1.1", "::1", b"192.0.2.1", None, ""])
def test_literals_and_empty_host_never_use_doh(monkeypatch, host):
    def fail(*args):
        raise socket.gaierror("no address")

    monkeypatch.setattr(net, "_original_getaddrinfo", fail)
    monkeypatch.setattr(net, "_doh_addresses", lambda *_: pytest.fail("DoH used"))
    with pytest.raises(socket.gaierror):
        net._getaddrinfo(host, 443)


def test_numeric_host_flag_never_uses_doh(monkeypatch):
    def fail(*args):
        raise socket.gaierror("numeric host required")

    monkeypatch.setattr(net, "_original_getaddrinfo", fail)
    monkeypatch.setattr(net, "_doh_addresses", lambda *_: pytest.fail("DoH used"))
    with pytest.raises(socket.gaierror):
        net._getaddrinfo("not-numeric", 443, flags=socket.AI_NUMERICHOST)


def test_doh_cache_cname_expiry_and_family(monkeypatch):
    calls = []
    clock = [0]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert kwargs["allow_redirects"] is False
            calls.append((url, kwargs["params"]))

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"Status": 0, "Answer": [
                        {"type": 5, "data": "alias.example", "TTL": 30},
                        {"type": 1, "data": "192.0.2.1", "TTL": 30},
                    ]}

            return Response()

    monkeypatch.setattr(net.requests, "Session", Session)
    monkeypatch.setattr(net.time, "monotonic", lambda: clock[0])
    assert net._doh_addresses("example", socket.AF_INET) == ("192.0.2.1",)
    clock[0] = 29
    assert net._doh_addresses("example", socket.AF_INET) == ("192.0.2.1",)
    assert len(calls) == 1
    clock[0] = 31
    net._doh_addresses("example", socket.AF_INET)
    assert len(calls) == 2


def test_google_backup(monkeypatch):
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            calls.append(url)
            if url == net._PROVIDERS[0]:
                raise requests.ConnectionError("offline")

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"Status": 0, "Answer": [
                        {"type": 28, "data": "2001:db8::1", "TTL": 600}
                    ]}

            return Response()

    monkeypatch.setattr(net.requests, "Session", Session)
    assert net._doh_addresses("example", socket.AF_INET6) == ("2001:db8::1",)
    assert calls == list(net._PROVIDERS)


@pytest.mark.parametrize("body", [
    [],
    {"Status": 3},
    {"Status": 0, "Answer": None},
    {"Status": 0, "Answer": ["malformed", {"type": 1, "data": "not-an-ip"}]},
])
def test_malformed_or_negative_doh_answers_fail_cleanly(monkeypatch, body):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return body

            return Response()

    monkeypatch.setattr(net.requests, "Session", Session)
    with pytest.raises(OSError, match="no addresses"):
        net._doh_addresses("example", socket.AF_INET)


def test_install_idempotent(monkeypatch):
    original = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", original)
    net.install_dns_fallback()
    first = socket.getaddrinfo
    net.install_dns_fallback()
    assert socket.getaddrinfo is first is net._getaddrinfo
