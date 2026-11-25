"""Checking a short-lived token another service signs for one person.

HS256 with a shared secret. The algorithm is fixed here rather than read from the
token, so a token can't choose how it gets checked. Errors name what was wrong and
never carry the token.

Whose token it is comes from the caller: the issuer and audience it must name, the
scopes there are, and a reader for the claims beyond the person and the scope
(agent_token.py is this service's).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Generic, TypeVar

import jwt

ALGORITHM = "HS256"
# Two machines' clocks disagree a little; a token lives for minutes.
LEEWAY_SECONDS = 30

T = TypeVar("T")


class InvalidToken(Exception):
    """Expired, forged, malformed or missing a claim."""


class TokenCheck(Generic[T]):
    """One issuer's tokens for one audience, read into the caller's own shape.

    `read_claims` gets the person, the scope and every claim, after the signature,
    the times, the issuer, the audience, the person and the scope have been checked;
    it returns the token as the caller knows it, or raises InvalidToken. A scope
    outside `scopes` is refused, so a check built with none refuses every token.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        scopes: frozenset[str],
        read_claims: Callable[[str, str, Mapping[str, object]], T],
    ) -> None:
        # The library skips a check it's given nothing for, so neither may be empty.
        if not issuer or not audience:
            raise ValueError("A token check needs an issuer and an audience.")
        self._issuer = issuer
        self._audience = audience
        self._scopes = frozenset(scopes)
        self._read_claims = read_claims

    def verify(self, raw: str, secrets: Sequence[str]) -> T:
        """The token's claims, or InvalidToken. Newest secret first."""
        claims = self._decode(raw, secrets)
        user_id, scope = claims["sub"], claims.get("scope")
        if not isinstance(user_id, str) or not user_id:
            raise InvalidToken("no identity")
        if not isinstance(scope, str) or scope not in self._scopes:
            raise InvalidToken("unknown scope")
        return self._read_claims(user_id, scope, claims)

    def _decode(self, raw: str, secrets: Sequence[str]) -> dict:
        last: jwt.InvalidTokenError = jwt.InvalidSignatureError("no secret")
        for secret in secrets:
            try:
                return jwt.decode(
                    raw,
                    secret,
                    algorithms=[ALGORITHM],
                    audience=self._audience,
                    issuer=self._issuer,
                    leeway=LEEWAY_SECONDS,
                    options={"require": ["exp", "iat", "sub", "aud", "iss"]},
                )
            except jwt.InvalidSignatureError as error:
                last = error  # Signed before a rotation, perhaps: try the next.
            except jwt.InvalidTokenError as error:
                raise InvalidToken(type(error).__name__) from None
        raise InvalidToken(type(last).__name__) from None
