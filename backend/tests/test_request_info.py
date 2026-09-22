from __future__ import annotations

import pytest

from app.core.request_info import resolve_client_ip


def test_forwarded_header_is_ignored_when_no_proxy_is_trusted() -> None:
    assert resolve_client_ip("10.0.0.5", "203.0.113.9", 0) == "10.0.0.5"


def test_single_trusted_proxy_uses_the_last_entry() -> None:
    assert resolve_client_ip("10.0.0.5", "203.0.113.9", 1) == "203.0.113.9"


def test_forged_left_hand_entries_are_ignored() -> None:
    # The client claims to be 1.2.3.4; our proxy appended the real address on the right.
    assert resolve_client_ip("10.0.0.5", "1.2.3.4, 203.0.113.9", 1) == "203.0.113.9"


def test_two_trusted_proxies_use_the_second_from_the_right() -> None:
    assert resolve_client_ip("10.0.0.5", "203.0.113.9, 10.1.1.1", 2) == "203.0.113.9"


@pytest.mark.parametrize("header", [None, "", "  ", "not-an-ip"])
def test_unusable_header_falls_back_to_the_socket_address(header) -> None:  # noqa: ANN001
    assert resolve_client_ip("10.0.0.5", header, 1) == "10.0.0.5"


def test_fewer_hops_than_trusted_proxies_is_not_trusted() -> None:
    # Two proxies are configured but only one hop is present: the header cannot be genuine.
    assert resolve_client_ip("10.0.0.5", "1.2.3.4", 2) == "10.0.0.5"


def test_missing_client_host_is_tolerated() -> None:
    assert resolve_client_ip(None, None, 0) is None


def test_ipv6_is_accepted() -> None:
    assert resolve_client_ip("10.0.0.5", "2001:db8::1", 1) == "2001:db8::1"
