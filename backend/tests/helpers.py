"""Shared helpers for API tests."""

from __future__ import annotations

from tests.factories import TEST_PASSWORD


def sign_in(client, user) -> None:  # noqa: ANN001
    response = client.post(
        "/api/auth/login", json={"email": user.email, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200, response.text


def upload(client, data: bytes, filename: str = "catalogue.pdf", **form: str):  # noqa: ANN001, ANN201
    return client.post(
        "/api/admin/catalogues",
        files={"file": (filename, data, "application/pdf")},
        data=form,
    )
