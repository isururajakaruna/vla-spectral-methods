"""Browser terminal (/terminal) and file browser (/browse) rooted at the project directory.

Both run shell commands or write files as the user running the studio, so they are served only
to requests made from the studio's own pages on a loopback address (see `request_allowed`).
The file browser can list, download, upload and create folders; it never deletes or overwrites.
"""

from __future__ import annotations

import base64
import fcntl
import ipaddress
import os
import pty
import shutil
import signal
import struct
import termios
import threading
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path.cwd(),
).resolve()
WEB_ASSETS = Path(__file__).with_name("assets") / "web"
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
TERMINAL_HISTORY_BYTES = 1 << 20
REQUEST_HEADER = "X-Studio-Request"


def is_loopback(host: str) -> bool:
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def request_allowed(handler: Any, is_post: bool) -> Optional[str]:
    """Return why a terminal/file request is refused, or None when it may proceed.

    The Host check defeats DNS rebinding, the Origin check rejects other sites' pages, and the
    custom header on writes forces a CORS preflight that this server never approves.
    """
    bound_host = handler.server.server_address[0]
    if not (is_loopback(bound_host) or getattr(handler.server, "allow_remote_tools", False)):
        return "The terminal and file browser are disabled when the studio listens on a non-loopback address"
    host_header = handler.headers.get("Host", "")
    hostname = urlsplit(f"//{host_header}").hostname or ""
    allowed = getattr(handler.server, "allowed_hosts", set())
    if not (is_loopback(hostname) or hostname == bound_host or hostname in allowed):
        return f"Unexpected Host header {hostname!r}; start the studio with --allowed-host {hostname} to trust it"
    origin = handler.headers.get("Origin")
    if origin and urlsplit(origin).netloc != host_header:
        return "Cross-origin requests are not allowed"
    if is_post and handler.headers.get(REQUEST_HEADER) != "1":
        return f"Missing {REQUEST_HEADER} header"
    return None


class TerminalSession:
    """One interactive shell on a pseudo-terminal, shared by every open /terminal page."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cond = threading.Condition()
        self._history = bytearray()
        self._total = 0
        self._pid: Optional[int] = None
        self._fd: Optional[int] = None
        self._size = (120, 32)

    def ensure_running(self) -> None:
        with self._cond:
            if self._pid is None:
                self._spawn()

    def _spawn(self) -> None:
        shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        pid, fd = pty.fork()
        if pid == 0:
            try:
                os.chdir(self.root)
                os.execvpe(shell, [shell, "-i"], env)
            finally:
                os._exit(127)
        self._pid, self._fd = pid, fd
        self._apply_size()
        threading.Thread(target=self._pump, args=(pid, fd), daemon=True).start()

    def _pump(self, pid: int, fd: int) -> None:
        while True:
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b""
            if not data:
                break
            self._append(data)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(fd)
        with self._cond:
            if self._pid == pid:
                self._pid, self._fd = None, None
        self._append(b"\r\n\x1b[2m[shell exited; press any key to start a new one]\x1b[0m\r\n")

    def _append(self, data: bytes) -> None:
        with self._cond:
            self._history += data
            self._total += len(data)
            if len(self._history) > TERMINAL_HISTORY_BYTES:
                del self._history[: len(self._history) - TERMINAL_HISTORY_BYTES]
            self._cond.notify_all()

    def read_since(self, offset: int, timeout: float) -> tuple[int, bytes]:
        """Output after absolute byte `offset` (from the oldest kept byte if it was trimmed)."""
        with self._cond:
            if offset > self._total:
                # The client's offset belongs to an earlier server process; replay what is kept.
                offset = 0
            self._cond.wait_for(lambda: self._total > offset, timeout=timeout)
            start = max(offset, self._total - len(self._history))
            return self._total, bytes(self._history[len(self._history) - (self._total - start) :])

    def write(self, data: bytes) -> None:
        self.ensure_running()
        with self._cond:
            fd = self._fd
        if fd is None:
            return
        try:
            os.write(fd, data)
        except OSError:
            pass

    def resize(self, cols: int, rows: int) -> None:
        self._size = (max(2, min(cols, 1000)), max(2, min(rows, 500)))
        with self._cond:
            self._apply_size()

    def _apply_size(self) -> None:
        if self._fd is None:
            return
        cols, rows = self._size
        try:
            # The kernel sends SIGWINCH to the foreground job when the size changes.
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass

    def stop(self) -> None:
        with self._cond:
            pid = self._pid
        if pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGHUP)
            except OSError:
                pass


TERMINAL = TerminalSession(PROJECT_ROOT)


def stream_terminal(handler: Any, offset: int) -> None:
    """Server-sent events carrying base64 terminal output; the event id is the byte offset."""
    TERMINAL.ensure_running()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache, no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        while True:
            offset, data = TERMINAL.read_since(offset, timeout=15.0)
            if data:
                payload = base64.b64encode(data).decode("ascii")
                handler.wfile.write(f"id: {offset}\ndata: {payload}\n\n".encode("ascii"))
            else:
                handler.wfile.write(b": keep-alive\n\n")
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return


class FileError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def resolve_in_root(relative: str) -> Path:
    """Resolve a project-relative path, refusing anything (including symlinks) outside the root."""
    if "\x00" in relative:
        raise FileError("Invalid path")
    target = (PROJECT_ROOT / relative.lstrip("/")).resolve()
    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise FileError("Path is outside the project", status=403)
    return target


def _relative(path: Path) -> str:
    return "" if path == PROJECT_ROOT else str(path.relative_to(PROJECT_ROOT))


def _check_new_name(name: str) -> list[str]:
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or "\x00" in p for p in parts):
        raise FileError(f"Invalid name {name!r}")
    return parts


def list_directory(relative: str) -> dict[str, Any]:
    directory = resolve_in_root(relative)
    if not directory.is_dir():
        raise FileError("Not a folder", status=404)
    entries = []
    for child in directory.iterdir():
        try:
            resolved = child.resolve()
            inside = resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents
            stat = child.stat()
        except OSError:
            continue
        if not inside:
            continue
        is_dir = child.is_dir()
        entries.append(
            {
                "name": child.name,
                "type": "dir" if is_dir else "file",
                "size": None if is_dir else stat.st_size,
                "mtime": stat.st_mtime,
                "symlink": child.is_symlink(),
            }
        )
    entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
    return {"root": str(PROJECT_ROOT), "path": _relative(directory), "entries": entries}


def send_download(handler: Any, relative: str) -> None:
    path = resolve_in_root(relative)
    if not path.is_file():
        raise FileError("Not a file", status=404)
    size = path.stat().st_size
    quoted = path.name.replace('"', "")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(size))
    handler.send_header("Content-Disposition", f'attachment; filename="{quoted}"')
    handler.end_headers()
    with path.open("rb") as f:
        shutil.copyfileobj(f, handler.wfile, 1 << 20)


def make_folder(relative_dir: str, name: str) -> dict[str, Any]:
    parent = resolve_in_root(relative_dir)
    if not parent.is_dir():
        raise FileError("Parent is not a folder", status=404)
    target = resolve_in_root(_relative(parent) + "/" + "/".join(_check_new_name(name)))
    if target.exists():
        raise FileError(f"{_relative(target)} already exists", status=409)
    target.mkdir(parents=True)
    return {"created": _relative(target)}


def receive_upload(handler: Any, relative_dir: str, name: str) -> dict[str, Any]:
    """Stream the request body into a new file; `name` may contain subfolders (folder uploads)."""
    parent = resolve_in_root(relative_dir)
    if not parent.is_dir():
        raise FileError("Upload folder does not exist", status=404)
    target = resolve_in_root(_relative(parent) + "/" + "/".join(_check_new_name(name)))
    if target.exists():
        raise FileError(f"{_relative(target)} already exists (the browser never overwrites files)", status=409)
    length = handler.headers.get("Content-Length")
    if length is None:
        raise FileError("Content-Length is required", status=411)
    remaining = int(length)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.upload-{uuid.uuid4().hex[:8]}")
    try:
        with staging.open("xb") as out:
            while remaining > 0:
                chunk = handler.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    raise FileError("Upload ended early")
                out.write(chunk)
                remaining -= len(chunk)
        # Hard-linking fails if the name appeared meanwhile, so an existing file is never replaced.
        os.link(staging, target)
    except FileExistsError as error:
        raise FileError(f"{_relative(target)} already exists", status=409) from error
    finally:
        staging.unlink(missing_ok=True)
    return {"uploaded": _relative(target), "bytes": int(length)}


def static_asset(name: str) -> Optional[tuple[Path, str]]:
    path = (WEB_ASSETS / name).resolve()
    if path.parent != WEB_ASSETS.resolve() or path.suffix not in STATIC_TYPES or not path.is_file():
        return None
    return path, STATIC_TYPES[path.suffix]


PAGE_STYLE = """
    :root { --bg:#f8f9fa; --card:#fff; --border:#dadce0; --text:#202124; --muted:#5f6368;
            --accent:#1a73e8; --accent-soft:#e8f0fe; --danger:#d93025; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Roboto, "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--text); }
    header { display:flex; align-items:center; gap:16px; padding:10px 20px; background:var(--card);
             border-bottom:1px solid var(--border); }
    header h1 { font-size:17px; font-weight:500; margin:0; }
    header nav { display:flex; gap:6px; margin-left:auto; }
    header nav a { color:var(--muted); text-decoration:none; font-size:13.5px; padding:6px 12px; border-radius:16px; }
    header nav a:hover { background:var(--accent-soft); color:var(--accent); }
    header nav a.active { background:var(--accent-soft); color:var(--accent); font-weight:500; }
    code, .mono { font-family: "Roboto Mono", Menlo, monospace; }
"""

NAV = """
  <nav>
    <a href="/">Studio</a>
    <a href="/terminal" class="{terminal}">Terminal</a>
    <a href="/browse" class="{browse}">Files</a>
    <a href="/experiments" class="{experiments}">Experiments</a>
  </nav>
"""

TERMINAL_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Terminal · Smooth Operator</title>
  <link rel="stylesheet" href="/static/xterm.css" />
  <style>
    __STYLE__
    html, body { height:100%; }
    body { display:flex; flex-direction:column; }
    .bar { display:flex; align-items:center; gap:10px; font-size:12.5px; color:var(--muted); }
    .dot { width:8px; height:8px; border-radius:50%; background:#f9ab00; display:inline-block; }
    .dot.live { background:#1e8e3e; }
    .dot.down { background:var(--danger); }
    #terminal-wrap { flex:1; min-height:0; padding:10px; background:#1e1e1e; }
    #terminal { height:100%; }
  </style>
</head>
<body>
  <header>
    <h1>Terminal</h1>
    <span class="bar"><span class="dot" id="conn-dot"></span><span id="conn-text">Connecting…</span>
      · <span class="mono">__ROOT__</span></span>
    __NAV__
  </header>
  <div id="terminal-wrap"><div id="terminal"></div></div>
  <script src="/static/xterm.js"></script>
  <script src="/static/addon-fit.js"></script>
  <script>
    const term = new Terminal({
      cursorBlink: true,
      fontFamily: '"Roboto Mono", Menlo, Consolas, monospace',
      fontSize: 13.5,
      scrollback: 20000,
      theme: { background: '#1e1e1e' },
    });
    const fit = new FitAddon.FitAddon();
    term.loadAddon(fit);
    term.open(document.getElementById('terminal'));
    fit.fit();
    term.focus();

    function post(url, body, type) {
      return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': type, 'X-Studio-Request': '1' },
        body,
      });
    }

    // Keystrokes are sent one request at a time, batching whatever was typed meanwhile, so
    // they always reach the shell in order.
    let pending = '';
    let sending = false;
    async function flushInput() {
      if (sending || !pending) return;
      sending = true;
      const chunk = pending;
      pending = '';
      try {
        await post('/api/terminal/input', new TextEncoder().encode(chunk), 'application/octet-stream');
      } catch (e) {
        pending = chunk + pending;
        await new Promise(r => setTimeout(r, 500));
      }
      sending = false;
      flushInput();
    }
    term.onData(data => { pending += data; flushInput(); });
    term.onBinary(data => { pending += data; flushInput(); });

    function sendSize() {
      post('/api/terminal/resize', JSON.stringify({ cols: term.cols, rows: term.rows }), 'application/json');
    }
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { fit.fit(); sendSize(); }, 80);
    });
    term.onResize(sendSize);
    sendSize();

    const dot = document.getElementById('conn-dot');
    const text = document.getElementById('conn-text');
    const source = new EventSource('/api/terminal/stream');
    source.onopen = () => { dot.className = 'dot live'; text.textContent = 'Live'; };
    source.onerror = () => { dot.className = 'dot down'; text.textContent = 'Reconnecting…'; };
    source.onmessage = (ev) => {
      const raw = atob(ev.data);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      term.write(bytes);
    };
  </script>
</body>
</html>
"""

BROWSE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Files · Smooth Operator</title>
  <style>
    __STYLE__
    main { max-width:1100px; margin:18px auto; padding:0 18px; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:12px; }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:12px 14px;
               border-bottom:1px solid var(--border); }
    .crumbs { display:flex; flex-wrap:wrap; align-items:center; gap:2px; font-size:14px; margin-right:auto; }
    .crumbs a { color:var(--accent); text-decoration:none; padding:3px 6px; border-radius:6px; cursor:pointer; }
    .crumbs a:hover { background:var(--accent-soft); }
    .crumbs span.sep { color:var(--muted); }
    button { font:inherit; font-size:13px; border:1px solid var(--border); background:#fff; color:var(--text);
             padding:6px 12px; border-radius:16px; cursor:pointer; }
    button:hover { background:var(--accent-soft); border-color:var(--accent); color:var(--accent); }
    table { width:100%; border-collapse:collapse; font-size:13.5px; }
    th { text-align:left; font-weight:500; color:var(--muted); font-size:12px; padding:8px 14px;
         border-bottom:1px solid var(--border); }
    td { padding:7px 14px; border-bottom:1px solid #f1f3f4; }
    tr:hover td { background:#f8f9fa; }
    td.size, td.time, th.size, th.time { text-align:right; white-space:nowrap; color:var(--muted); }
    .name a { color:var(--text); text-decoration:none; cursor:pointer; }
    .name a:hover { color:var(--accent); text-decoration:underline; }
    .icon { display:inline-block; width:22px; }
    .empty { padding:30px; text-align:center; color:var(--muted); }
    #drop { margin:12px 14px; padding:14px; border:2px dashed var(--border); border-radius:10px;
            text-align:center; color:var(--muted); font-size:13px; }
    #drop.over { border-color:var(--accent); background:var(--accent-soft); color:var(--accent); }
    #uploads { padding:0 14px; font-size:12.5px; }
    .up { display:flex; gap:10px; align-items:center; padding:3px 0; }
    .up .bar { flex:0 0 160px; height:6px; background:#e8eaed; border-radius:3px; overflow:hidden; }
    .up .bar > div { height:100%; width:0; background:var(--accent); }
    .up.err { color:var(--danger); }
    .note { font-size:12px; color:var(--muted); padding:8px 14px 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Files</h1>
    <span class="mono" style="font-size:12.5px;color:var(--muted);">__ROOT__</span>
    __NAV__
  </header>
  <main>
    <div class="card">
      <div class="toolbar">
        <div class="crumbs" id="crumbs"></div>
        <button type="button" onclick="newFolder()">New folder</button>
        <button type="button" onclick="document.getElementById('pick-files').click()">Upload files</button>
        <button type="button" onclick="document.getElementById('pick-folder').click()">Upload folder</button>
        <input type="file" id="pick-files" multiple hidden />
        <input type="file" id="pick-folder" webkitdirectory multiple hidden />
      </div>
      <div id="drop">Drop files or folders here to upload them into this folder</div>
      <div id="uploads"></div>
      <table>
        <thead><tr><th>Name</th><th class="size">Size</th><th class="time">Modified</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="note">Existing files are never overwritten, and nothing can be deleted from this page.</div>
    </div>
  </main>
  <script>
    let currentPath = new URLSearchParams(location.search).get('path') || '';

    const esc = (s) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const join = (dir, name) => (dir ? dir + '/' : '') + name;
    function fmtSize(n) {
      if (n == null) return '';
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      let i = 0;
      while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
      return (i ? n.toFixed(1) : n) + ' ' + units[i];
    }

    async function load(path, push = true) {
      const res = await fetch('/api/files/list?path=' + encodeURIComponent(path));
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      currentPath = data.path;
      if (push) history.pushState({ path: currentPath }, '', '/browse' + (currentPath ? '?path=' + encodeURIComponent(currentPath) : ''));

      const crumbs = document.getElementById('crumbs');
      const parts = currentPath ? currentPath.split('/') : [];
      let html = `<a data-path="">${esc(data.root.split('/').pop() || '/')}</a>`;
      parts.forEach((p, i) => {
        html += `<span class="sep">/</span><a data-path="${esc(parts.slice(0, i + 1).join('/'))}">${esc(p)}</a>`;
      });
      crumbs.innerHTML = html;
      crumbs.querySelectorAll('a').forEach(a => { a.onclick = () => load(a.dataset.path); });

      const rows = document.getElementById('rows');
      if (!data.entries.length) {
        rows.innerHTML = '<tr><td colspan="3" class="empty">This folder is empty</td></tr>';
        return;
      }
      rows.innerHTML = data.entries.map(e => {
        const rel = join(currentPath, e.name);
        const link = e.type === 'dir'
          ? `<a data-dir="${esc(rel)}">${esc(e.name)}/</a>`
          : `<a href="/api/files/download?path=${encodeURIComponent(rel)}" download>${esc(e.name)}</a>`;
        return `<tr><td class="name"><span class="icon">${e.type === 'dir' ? '📁' : '📄'}</span>${link}${e.symlink ? ' <span style="color:var(--muted)">↪</span>' : ''}</td>`
          + `<td class="size">${fmtSize(e.size)}</td><td class="time">${new Date(e.mtime * 1000).toLocaleString()}</td></tr>`;
      }).join('');
      rows.querySelectorAll('a[data-dir]').forEach(a => { a.onclick = () => load(a.dataset.dir); });
    }
    window.onpopstate = (ev) => load((ev.state && ev.state.path) || '', false);

    async function newFolder() {
      const name = prompt('New folder name (use a/b to create nested folders):');
      if (!name) return;
      const res = await fetch('/api/files/mkdir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Studio-Request': '1' },
        body: JSON.stringify({ dir: currentPath, name }),
      });
      const data = await res.json();
      if (data.error) alert(data.error);
      load(currentPath, false);
    }

    function uploadOne(file, relName, dir) {
      const row = document.createElement('div');
      row.className = 'up';
      row.innerHTML = `<div class="bar"><div></div></div><span>${esc(join(dir, relName))}</span><span class="st"></span>`;
      document.getElementById('uploads').appendChild(row);
      return new Promise(resolve => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `/api/files/upload?dir=${encodeURIComponent(dir)}&name=${encodeURIComponent(relName)}`);
        xhr.setRequestHeader('X-Studio-Request', '1');
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) row.querySelector('.bar > div').style.width = (100 * ev.loaded / ev.total) + '%';
        };
        xhr.onload = () => {
          let data = {};
          try { data = JSON.parse(xhr.responseText); } catch (e) {}
          if (xhr.status >= 400 || data.error) {
            row.classList.add('err');
            row.querySelector('.st').textContent = data.error || `HTTP ${xhr.status}`;
          } else {
            row.querySelector('.bar > div').style.width = '100%';
            row.querySelector('.st').textContent = 'done';
            setTimeout(() => row.remove(), 4000);
          }
          resolve();
        };
        xhr.onerror = () => { row.classList.add('err'); row.querySelector('.st').textContent = 'network error'; resolve(); };
        xhr.send(file);
      });
    }

    async function uploadAll(items) {
      const dir = currentPath;
      for (const { file, name } of items) await uploadOne(file, name, dir);
      load(dir, false);
    }

    const filesOf = (input) => [...input.files].map(f => ({ file: f, name: f.webkitRelativePath || f.name }));
    document.getElementById('pick-files').onchange = (ev) => { uploadAll(filesOf(ev.target)); ev.target.value = ''; };
    document.getElementById('pick-folder').onchange = (ev) => { uploadAll(filesOf(ev.target)); ev.target.value = ''; };

    // Dropped folders are walked recursively so their structure is recreated.
    async function walk(entry, prefix, out) {
      if (entry.isFile) {
        const file = await new Promise((res, rej) => entry.file(res, rej));
        out.push({ file, name: prefix + entry.name });
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          for (const child of batch) await walk(child, prefix + entry.name + '/', out);
        } while (batch.length);
      }
    }
    const drop = document.getElementById('drop');
    drop.ondragover = (ev) => { ev.preventDefault(); drop.classList.add('over'); };
    drop.ondragleave = () => drop.classList.remove('over');
    drop.ondrop = async (ev) => {
      ev.preventDefault();
      drop.classList.remove('over');
      const entries = [...ev.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
      const items = [];
      if (entries.length) {
        for (const entry of entries) await walk(entry, '', items);
      } else {
        for (const f of ev.dataTransfer.files) items.push({ file: f, name: f.name });
      }
      uploadAll(items);
    };

    load(currentPath, false);
  </script>
</body>
</html>
"""


def render_page(page: str, active: str) -> str:
    nav = NAV
    for name in ("terminal", "browse", "experiments"):
        nav = nav.replace("{" + name + "}", "active" if active == name else "")
    root = str(PROJECT_ROOT).replace("&", "&amp;").replace("<", "&lt;")
    return page.replace("__STYLE__", PAGE_STYLE).replace("__NAV__", nav).replace("__ROOT__", root)
