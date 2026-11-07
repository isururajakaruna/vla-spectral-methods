import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from smooth_operator import devtools


class FakeHandler:
    def __init__(self, headers, body=b"", bound="127.0.0.1"):
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.server = SimpleNamespace(server_address=(bound, 8765))


class FileBrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        patcher = mock.patch.object(devtools, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_paths_cannot_leave_the_root(self):
        (self.root / "link").symlink_to("/etc")
        for path in ("..", "../x", "/../../etc/passwd", "link/passwd"):
            with self.assertRaises(devtools.FileError):
                devtools.resolve_in_root(path)
        self.assertNotIn("link", [e["name"] for e in devtools.list_directory("")["entries"]])

    def test_upload_creates_folders_and_never_overwrites(self):
        handler = FakeHandler({"Content-Length": "5"}, b"hello")
        devtools.receive_upload(handler, "", "a/b/c.txt")
        self.assertEqual((self.root / "a/b/c.txt").read_bytes(), b"hello")
        with self.assertRaises(devtools.FileError):
            devtools.receive_upload(FakeHandler({"Content-Length": "3"}, b"new"), "", "a/b/c.txt")
        self.assertEqual((self.root / "a/b/c.txt").read_bytes(), b"hello")
        with self.assertRaises(devtools.FileError):
            devtools.receive_upload(FakeHandler({"Content-Length": "1"}, b"x"), "", "../escape.txt")
        self.assertEqual(sorted(p.name for p in (self.root / "a/b").iterdir()), ["c.txt"])

    def test_make_folder_refuses_existing(self):
        devtools.make_folder("", "x/y")
        self.assertTrue((self.root / "x/y").is_dir())
        with self.assertRaises(devtools.FileError):
            devtools.make_folder("", "x/y")


class RequestGuardTests(unittest.TestCase):
    def test_guard(self):
        ok = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765", "X-Studio-Request": "1"}
        self.assertIsNone(devtools.request_allowed(FakeHandler(ok), is_post=True))
        self.assertIsNotNone(devtools.request_allowed(FakeHandler({**ok, "Origin": "http://evil.test"}), is_post=True))
        self.assertIsNotNone(devtools.request_allowed(FakeHandler({**ok, "Host": "evil.test:8765"}), is_post=False))
        self.assertIsNotNone(devtools.request_allowed(FakeHandler({"Host": "127.0.0.1:8765"}), is_post=True))
        self.assertIsNotNone(devtools.request_allowed(FakeHandler(ok, bound="0.0.0.0"), is_post=False))
        tunnel = {"Host": "demo.ngrok-free.dev", "Origin": "https://demo.ngrok-free.dev", "X-Studio-Request": "1"}
        handler = FakeHandler(tunnel)
        self.assertIsNotNone(devtools.request_allowed(handler, is_post=True))
        handler.server.allowed_hosts = {"demo.ngrok-free.dev"}
        self.assertIsNone(devtools.request_allowed(handler, is_post=True))


if __name__ == "__main__":
    unittest.main()
