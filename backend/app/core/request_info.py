"""Client-address resolution. Pure logic, separated from the web framework."""

from __future__ import annotations

import ipaddress


def resolve_client_ip(
    client_host: str | None, forwarded_for: str | None, trusted_proxies: int
) -> str | None:
    """Best-effort real client IP.

    ``X-Forwarded-For`` is attacker-controlled unless it was written by a proxy we run,
    so it is ignored unless ``trusted_proxies`` > 0. With N trusted proxies the address
    appended by the outermost one is the N-th entry from the right; anything to its left
    could have been forged by the client.
    """
    if trusted_proxies <= 0 or not forwarded_for:
        return client_host
    hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
    if len(hops) < trusted_proxies:
        return client_host  # fewer hops than expected: do not trust the header
    candidate = hops[-trusted_proxies]
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return client_host
    return candidate
