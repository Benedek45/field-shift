"""System-first DNS with a verified HTTPS fallback for broken local resolvers."""

import ipaddress
import logging
import socket
import threading
import time

import requests

_LOG = logging.getLogger(__name__)
_PROVIDERS = ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve")
_original_getaddrinfo = socket.getaddrinfo
_lock = threading.RLock()
_cache = {}
_warned = set()


def _doh_addresses(host, family):
    record_types = {
        socket.AF_UNSPEC: (1, 28),
        socket.AF_INET: (1,),
        socket.AF_INET6: (28,),
    }.get(family, ())
    if not record_types:
        raise ValueError(f"Unsupported address family: {family}")
    key = (host, family)
    with _lock:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
    for provider in _PROVIDERS:
        addresses = []
        ttls = []
        # Ignore proxy environment settings: the bootstrap must connect to the IP.
        with requests.Session() as session:
            session.trust_env = False
            for record_type in record_types:
                try:
                    response = session.get(
                        provider,
                        params={"name": host, "type": record_type},
                        headers={"Accept": "application/dns-json"},
                        timeout=(5, 10),
                        allow_redirects=False,
                    )
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict) or body.get("Status") != 0:
                        continue
                    for answer in body.get("Answer", []):
                        if not isinstance(answer, dict):
                            continue
                        if answer.get("type") != record_type:
                            continue
                        address = ipaddress.ip_address(answer["data"])
                        if address.version != (4 if record_type == 1 else 6):
                            continue
                        addresses.append(str(address))
                        ttls.append(max(0, int(answer.get("TTL", 60))))
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    continue
        if addresses:
            addresses = tuple(dict.fromkeys(addresses))
            with _lock:
                _cache[key] = (time.monotonic() + min(300, *ttls), addresses)
            return addresses
    raise OSError(f"DNS-over-HTTPS returned no addresses for {host}")


def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror as original_error:
        if not isinstance(host, (str, bytes)) or not host:
            raise
        try:
            name = host.decode("ascii") if isinstance(host, bytes) else host
            ipaddress.ip_address(name)
        except ValueError:
            pass
        else:
            raise
        if flags & socket.AI_NUMERICHOST:
            raise
        try:
            addresses = _doh_addresses(name.encode("idna").decode("ascii"), family)
            result = []
            for address in addresses:
                result.extend(
                    _original_getaddrinfo(address, port, family, type, proto, flags)
                )
            if not result:
                raise OSError("No usable fallback addresses")
        except (OSError, ValueError, UnicodeError):
            raise original_error from None
        with _lock:
            if name not in _warned:
                _LOG.warning("System DNS failed for %s; using DNS-over-HTTPS", name)
                _warned.add(name)
        return result


def install_dns_fallback():
    """Install once; successful system lookups and hostname-based TLS are unchanged."""
    with _lock:
        if socket.getaddrinfo is not _getaddrinfo:
            socket.getaddrinfo = _getaddrinfo
