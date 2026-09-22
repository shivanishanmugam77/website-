"""The three plain HTML pages that make up the start of the admin website (Phase 7): sign in,
the catalogue dashboard, and one catalogue's page. These pages are shells - real access
control lives in the JSON APIs their own JavaScript calls, so the pages themselves load for
anyone and only fail (via a client-side redirect to sign in) once that JavaScript runs."""

from __future__ import annotations

import uuid

from tests.factories import TEST_PASSWORD, make_catalogue, make_user
from tests.helpers import sign_in


def assert_page(response, contains: str | None = None) -> None:  # noqa: ANN001
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "content-security-policy" in response.headers
    if contains:
        assert contains in response.text


def test_the_login_page_loads_for_anyone(client) -> None:  # noqa: ANN001
    assert_page(client.get("/api/admin/login"))


def test_the_dashboard_loads_even_when_signed_out(client) -> None:  # noqa: ANN001
    """The page itself is just a shell; its own fetch calls are what actually enforce
    sign-in, redirecting to /login rather than showing a raw JSON error."""
    assert_page(client.get("/api/admin/dashboard"))


def test_the_dashboard_loads_when_signed_in(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    sign_in(client, user)
    assert_page(client.get("/api/admin/dashboard"))


def test_a_catalogues_page_embeds_its_own_id(client, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    assert_page(client.get(f"/api/admin/catalogues/{catalogue.id}/dashboard"), contains=str(catalogue.id))


def test_an_unknown_catalogues_page_is_not_found(client) -> None:  # noqa: ANN001
    response = client.get(f"/api/admin/catalogues/{uuid.uuid4()}/dashboard")
    assert response.status_code == 404
