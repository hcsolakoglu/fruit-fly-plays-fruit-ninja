"""Sharing must work after copying the address into a cookie-less browser."""

import http.client
import json
from pathlib import Path
import secrets
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from share_demo import ShareHandler, ThreadingHTTPServer  # noqa: E402


class SharingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class TestHandler(ShareHandler):
            token = secrets.token_urlsafe(32)
            expires_at = time.time() + 3600

        cls.handler = TestHandler
        cls.prefix = f"/invite/{TestHandler.token}/"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, method="GET", body=None, headers=None):
        # Every call starts fresh, without any browser cookie state.
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=10)
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        result = response.status, dict(response.headers), response.read()
        conn.close()
        return result

    def test_address_bar_keeps_invitation_and_works_without_cookies(self):
        status, headers, _ = self.request(self.prefix.rstrip("/"))
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], self.prefix + "#experiment")
        status, headers, body = self.request(headers["Location"].split("#")[0])
        self.assertEqual(status, 200)
        self.assertIn(f'<base href="{self.prefix}">'.encode(), body)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(self.request(self.prefix + "web/app.js")[0], 200)
        self.assertEqual(self.request(self.prefix + "data/scene3d.json")[0], 200)
        self.assertEqual(self.request(self.prefix + "api/config")[0], 200)

    def test_live_actions_work_without_cookies(self):
        headers = {"Content-Type": "application/json"}
        status, _, body = self.request(
            self.prefix + "api/reset", "POST", '{"seed":7}', headers
        )
        self.assertEqual(status, 200)
        frame = json.loads(body)["state"]["frame"]
        status, _, body = self.request(self.prefix + "api/step", "POST", "{}", headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"]["frame"], frame + 1)

    def test_bad_or_absent_invitation_and_repository_paths_stay_blocked(self):
        self.assertEqual(self.request("/")[0], 403)
        self.assertEqual(self.request("/invite/wrong/")[0], 403)
        for path in [
            ".git/config",
            "videos/",
            "data/connectome.npz",
            "web/../.git/config",
            "web/%2e%2e/.git/config",
        ]:
            for method in ["GET", "HEAD"]:
                self.assertEqual(self.request(self.prefix + path, method)[0], 404)

    def test_expired_invitation_is_denied(self):
        old_expiry = self.handler.expires_at
        try:
            self.handler.expires_at = time.time() - 1
            self.assertEqual(self.request(self.prefix)[0], 410)
        finally:
            self.handler.expires_at = old_expiry

    def test_video_ranges_work_without_cookies(self):
        status, headers, body = self.request(
            self.prefix + "videos/06_fly_at_pc_3d.mp4",
            headers={"Range": "bytes=0-1023"},
        )
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 1024)
        self.assertEqual(headers["Accept-Ranges"], "bytes")

    def test_old_cookie_links_redirect_to_shareable_address(self):
        status, headers, _ = self.request(
            "/", headers={"Cookie": "fly_demo=" + self.handler.token}
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], self.prefix + "#experiment")

    def test_cross_origin_actions_are_denied(self):
        status, _, _ = self.request(
            self.prefix + "api/step",
            "POST",
            "{}",
            {"Content-Type": "application/json", "Origin": "https://unrelated.invalid"},
        )
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
