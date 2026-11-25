"""Whether a person's MCP server may be reached from here.

This process can reach what a person can't: the code runner, the database, cloud
metadata. So a server's address must be public when this service connects, whatever
it resolved to when it was saved. A hosted toolkit's mount is checked by its own host list.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any, Optional
from urllib.parse import urlparse

import httpcore


# IPv6 forms that carry an IPv4 address, which a network may route to that address.
# They have no use for a public server, so they're refused rather than unwrapped.
_EMBEDS_IPV4 = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0:0/96"),
    ipaddress.ip_network("64:ff9b::/96"),
)


class ServerAddressRefused(Exception):
    """The server's address isn't one this service may connect to."""


def _is_public(address: str) -> bool:
    # A scoped address such as fe80::1%en0 is local by definition.
    if "%" in address:
        return False
    ip = ipaddress.ip_address(address)
    if ip.version == 6 and any(ip in network for network in _EMBEDS_IPV4):
        return False
    target = getattr(ip, "ipv4_mapped", None) or ip
    # Python calls multicast and the old site-local range (fec0::/10) global.
    return (
        target.is_global
        and not target.is_multicast
        and not getattr(target, "is_site_local", False)
    )


async def check_server_address(url: str, *, allow_local: bool = False) -> None:
    """Raises ServerAddressRefused unless every address for the URL's host is public.
    ``allow_local`` is the development setting (mcp_limits.py): anything goes."""
    if allow_local:
        return
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ServerAddressRefused("not an https address")
    try:
        found = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname, parsed.port or 443
        )
    except OSError as error:
        raise ServerAddressRefused("host not found") from error
    if not found or not all(_is_public(info[4][0]) for info in found):
        raise ServerAddressRefused("host isn't public")


class PublicPeersOnly(httpcore.AsyncNetworkBackend):
    """Refuses a connection whose other end isn't public, before anything is sent.

    The check above looks the name up once, and the HTTP client looks it up again,
    so a name can answer differently the second time. This looks at the socket
    itself, on every connection a toolset opens, however long it has been cached.
    """

    def __init__(self, *, allow_local: bool = False) -> None:
        self._backend = httpcore.AnyIOBackend()
        self._allow_local = allow_local

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._backend.connect_tcp(
            host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        if self._allow_local:
            return stream
        peer = stream.get_extra_info("server_addr")
        if not peer or not _is_public(peer[0]):
            await stream.aclose()
            raise httpcore.ConnectError("server address isn't public")
        return stream

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)
