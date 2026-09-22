from __future__ import annotations

import hashlib

import pytest

from app.services.storage import (
    InvalidStorageKeyError,
    LocalStorage,
    ObjectNotFoundError,
    ObjectTooLargeError,
    validate_key,
)


@pytest.fixture
def storage(tmp_path) -> LocalStorage:  # noqa: ANN001
    return LocalStorage(tmp_path / "root")


def test_round_trip_and_metadata(storage: LocalStorage) -> None:
    stored = storage.put_bytes("catalogues/abc/original.pdf", b"hello world")
    assert stored.size == 11
    assert stored.sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert storage.read_bytes("catalogues/abc/original.pdf") == b"hello world"
    assert storage.exists("catalogues/abc/original.pdf")
    assert storage.size("catalogues/abc/original.pdf") == 11


def test_streaming_upload_hashes_incrementally(storage: LocalStorage) -> None:
    chunks = [b"a" * 1000, b"b" * 1000, b"c"]
    stored = storage.put_stream("k/file.bin", iter(chunks))
    assert stored.size == 2001
    assert stored.sha256 == hashlib.sha256(b"".join(chunks)).hexdigest()


def test_oversized_stream_is_rejected_and_leaves_no_debris(storage: LocalStorage, tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(ObjectTooLargeError):
        storage.put_stream("k/big.bin", iter([b"x" * 600, b"x" * 600]), max_bytes=1000)
    assert not storage.exists("k/big.bin")
    leftovers = [p for p in (tmp_path / "root").rglob("*") if p.is_file()]
    assert leftovers == []


def test_failed_stream_leaves_no_partial_file(storage: LocalStorage, tmp_path) -> None:  # noqa: ANN001
    def exploding():  # noqa: ANN202
        yield b"partial"
        raise RuntimeError("connection dropped")

    with pytest.raises(RuntimeError):
        storage.put_stream("k/half.bin", exploding())
    assert [p for p in (tmp_path / "root").rglob("*") if p.is_file()] == []


def test_overwrite_is_atomic_replacement(storage: LocalStorage) -> None:
    storage.put_bytes("k/f.txt", b"old")
    storage.put_bytes("k/f.txt", b"new")
    assert storage.read_bytes("k/f.txt") == b"new"


def test_missing_objects(storage: LocalStorage) -> None:
    assert not storage.exists("nope/file.bin")
    with pytest.raises(ObjectNotFoundError):
        storage.read_bytes("nope/file.bin")
    with pytest.raises(ObjectNotFoundError):
        storage.size("nope/file.bin")
    with pytest.raises(ObjectNotFoundError), storage.local_copy("nope/file.bin"):
        pass
    storage.delete("nope/file.bin")  # deleting a missing object is not an error


def test_local_copy_gives_a_real_readable_path(storage: LocalStorage) -> None:
    storage.put_bytes("k/doc.pdf", b"%PDF-1.4")
    with storage.local_copy("k/doc.pdf") as path:
        assert path.read_bytes() == b"%PDF-1.4"


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "/etc/passwd",
        "../outside.txt",
        "a/../../outside.txt",
        "a/./b",
        "a//b",
        ".hidden",
        "a/.hidden/b",
        "a\\b",
        "a/b\x00.txt",
        "a b/file.txt",
        "a/b/",
        "C:/windows",
        "x" * 600,
    ],
)
def test_dangerous_keys_are_rejected_everywhere(storage: LocalStorage, bad_key: str) -> None:
    with pytest.raises(InvalidStorageKeyError):
        validate_key(bad_key)
    with pytest.raises(InvalidStorageKeyError):
        storage.put_bytes(bad_key, b"x")
    with pytest.raises(InvalidStorageKeyError):
        storage.read_bytes(bad_key)
    with pytest.raises(InvalidStorageKeyError):
        storage.delete(bad_key)
    with pytest.raises(InvalidStorageKeyError):
        storage.delete_prefix(bad_key)


def test_symlink_escape_is_blocked(storage: LocalStorage, tmp_path) -> None:  # noqa: ANN001
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    link = tmp_path / "root" / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidStorageKeyError):
        storage.read_bytes("link/secret.txt")


def test_delete_prefix_removes_only_that_subtree(storage: LocalStorage) -> None:
    storage.put_bytes("catalogues/one/original.pdf", b"1")
    storage.put_bytes("catalogues/one/pages/0001.jpg", b"2")
    storage.put_bytes("catalogues/one/pages/0002.jpg", b"3")
    storage.put_bytes("catalogues/two/original.pdf", b"4")
    assert storage.delete_prefix("catalogues/one") == 3
    assert not storage.exists("catalogues/one/original.pdf")
    assert storage.exists("catalogues/two/original.pdf")
    assert storage.delete_prefix("catalogues/one") == 0  # idempotent
    assert storage.delete_prefix("never/existed") == 0
