"""Invite-only demo server for a temporary HTTPS tunnel. No repository browsing."""

import argparse
import hmac
import io
from http.cookies import CookieError, SimpleCookie
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit

from serve import Handler, ROOT


class ShareHandler(Handler):
    token = ""
    expires_at = 0
    allowed_files = {
        "/",
        "/web/index.html",
        "/web/app.js",
        "/web/scene3d.js",
        "/data/scene3d.json",
        "/README.md",
        "/docs/RESULTS.md",
        "/docs/SOURCES.md",
        "/docs/EXPERIMENT.md",
        "/docs/POSTS.md",
    } | {
        "/" + str(p.relative_to(ROOT))
        for p in (ROOT / "videos").iterdir()
        if p.suffix in {".mp4", ".jpg"} and p.is_file() and not p.is_symlink()
    }

    def end_headers(self):
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def authorized(self, supplied_token=None):
        if time.time() >= self.expires_at:
            self.send_error(410, "This temporary invitation has expired")
            return False
        if supplied_token is not None:
            if hmac.compare_digest(supplied_token.encode(), self.token.encode()):
                return True
            self.send_error(403, "Invalid invitation")
            return False
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            supplied = cookies.get("fly_demo")
            if supplied and hmac.compare_digest(supplied.value, self.token):
                return True
        except (CookieError, TypeError):
            pass
        self.send_error(403, "Private demo: open the full invitation link you received")
        return False

    def resolve_invitation(self):
        """Keep authorization in the shareable URL; cookies are only a fallback."""
        path = urlsplit(self.path).path
        self.share_prefix = f"/invite/{self.token}/"
        if path.startswith("/invite/"):
            token, slash, relative = path[8:].partition("/")
            if not self.authorized(token):
                return False
            if not slash:
                self.redirect_to_invitation()
                return False
            self.path = "/" + relative
            return True
        if not self.authorized():
            return False
        self.path = path
        if path in {"/", "/web/index.html"}:
            self.redirect_to_invitation()
            return False
        return True

    def redirect_to_invitation(self):
        self.send_response(303)
        ttl = max(0, int(self.expires_at - time.time()))
        self.send_header(
            "Set-Cookie",
            f"fly_demo={self.token}; Path=/; Max-Age={ttl}; Secure; HttpOnly; SameSite=Lax",
        )
        self.send_header("Location", self.share_prefix + "#experiment")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def read_request(self, head=False):
        if not self.resolve_invitation():
            return
        path = self.path
        if path not in self.allowed_files and path != "/api/config":
            self.send_error(404)
            return
        self.path = path
        if head:
            if path == "/":
                self.path = "/web/index.html"
            if path == "/api/config":
                self.send_error(405)
            else:
                super().do_HEAD()
        else:
            super().do_GET()

    def do_GET(self):
        self.read_request()

    def do_HEAD(self):
        self.read_request(head=True)

    def send_head(self):
        if self.path == "/web/index.html":
            html = (
                (ROOT / "web/index.html")
                .read_text()
                .replace('<base href="/">', f'<base href="{self.share_prefix}">')
                .encode()
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            return io.BytesIO(html)
        if not self.path.endswith(".mp4"):
            return super().send_head()
        path = ROOT / self.path.lstrip("/")
        size = path.stat().st_size
        start, end = 0, size - 1
        partial = self.headers.get("Range")
        if partial:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", partial)
            if not match or not any(match.groups()):
                self.send_error(416)
                return None
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), end) if right else end
            else:
                start = max(0, size - int(right))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        stream = path.open("rb")
        stream.seek(start)
        self._remaining_video_bytes = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        if not hasattr(self, "_remaining_video_bytes"):
            return super().copyfile(source, outputfile)
        remaining = self._remaining_video_bytes
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def do_POST(self):
        if not self.resolve_invitation():
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(403, "Cross-origin request denied")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self.send_error(415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 1024:
                self.send_error(413)
                return
            super().do_POST()
        except (ValueError, TypeError, KeyError):
            self.send_error(400)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--invite-file", type=Path, required=True)
    a = p.parse_args()
    ShareHandler.token = secrets.token_urlsafe(32)
    ShareHandler.expires_at = time.time() + a.hours * 3600
    server = ThreadingHTTPServer(("127.0.0.1", a.port), ShareHandler)
    fd = os.open(a.invite_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("/invite/" + ShareHandler.token + "\n")
    print(
        f"Invite-only demo listening on 127.0.0.1:{a.port}; expires in {a.hours:g} hours",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
