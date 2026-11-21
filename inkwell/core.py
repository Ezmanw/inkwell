"""Inkwell backend: runs llama-server, exposes a key-protected OpenAI-compatible
API on the LAN (for phone apps), and connects dsh.

Ports:
  8686  llama-server (this PC only, no key)  -> dsh / local apps
  8691  LAN API gateway (only while Wi-Fi sharing is on; Bearer key required)
"""
import json, os, re, secrets, shutil, signal, socket, subprocess, threading, time
import urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

DATA = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "inkwell"
SHARE = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "inkwell"
DATA.mkdir(parents=True, exist_ok=True)
CONF_PATH = DATA / "config.json"
CHATS_PATH = DATA / "chats.json"
LOG_PATH = DATA / "llama-server.log"
DSH_HOME = Path(os.environ.get("DSH_HOME", Path.home() / ".dsh"))

LLAMA_PORT, LAN_PORT = 8686, 8691
MODEL_ID = "inkling-2-mini"

DEFAULTS = {
    "setup_done": False,
    "backend": "",
    "llama_mode": "recommended",          # recommended (tested + VRAM fix) | latest
    "llama_src": str(SHARE / "llama.cpp"),
    "llama_server": "",
    "models_dir": str(SHARE / "models"),
    "model": "",
    "mmproj": "",
    "cpu_layers": 0,                       # how many non-loop layers' experts live in system RAM
    "vision_mode": "cpu",  # gpu | cpu | off
    "ctx": 32768,
    "ubatch": 128,
    "kv_quant": "q8_0",
    "gpu_layers": 99,
    "threads": max(1, (os.cpu_count() or 2) // 2),
    "extra_args": "",
    "lan": False,
    "token": "sk-inkwell-" + secrets.token_urlsafe(18),
    "dsh_profile": "web",
    "autostart": True,
    "think_effort": 0.7,     # 0 (off) .. 0.99 (max); default for every client (chat, dsh, phone)
    "think_budget": -1,      # max thinking tokens, -1 = no limit
    "think_preserve": True,  # keep earlier turns' thinking in the context
}


def load_conf():
    conf = dict(DEFAULTS)
    if CONF_PATH.exists():
        try:
            conf.update(json.loads(CONF_PATH.read_text()))
        except Exception:
            pass
    if "vision" in conf:  # old on/off setting
        if conf.pop("vision") is False:
            conf["vision_mode"] = "off"
    return conf


def save_conf():
    CONF_PATH.write_text(json.dumps(CONF, indent=2))
    os.chmod(CONF_PATH, 0o600)


CONF = load_conf()
save_conf()


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))  # UDP connect sends nothing; just picks the LAN interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------- llama-server process ----------------
_lock = threading.RLock()
_proc = None


def llama_cmd():
    c = CONF
    cmd = [c["llama_server"], "-m", c["model"], "--alias", MODEL_ID,
           "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
           "-ngl", str(c["gpu_layers"]), "-c", str(c["ctx"]),
           "-ub", str(c["ubatch"]), "-b", str(max(int(c["ubatch"]), 512)),
           "-fa", "on", "--jinja", "-np", "1", "-t", str(c["threads"]),
           "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
           "--no-webui",
           "--chat-template-kwargs", json.dumps({"reasoning_effort": round(float(c["think_effort"]), 2)})]
    if int(c["think_budget"]) >= 0:
        cmd += ["--reasoning-budget", str(int(c["think_budget"])),
                "--reasoning-budget-message", "\n\nI've thought about this enough. Time to answer.\n"]
    if not c["think_preserve"]:
        cmd += ["--no-reasoning-preserve"]
    if c["vision_mode"] != "off" and Path(c["mmproj"]).exists():
        cmd += ["--mmproj", c["mmproj"]]
        if c["vision_mode"] == "cpu":
            cmd += ["--no-mmproj-offload"]
    else:
        cmd += ["--no-mmproj"]
    from .catalog import offload_args
    cmd += offload_args(int(c.get("cpu_layers", 0)))
    if c["kv_quant"] != "f16":
        cmd += ["-ctk", c["kv_quant"], "-ctv", c["kv_quant"]]
    if c["extra_args"].strip():
        cmd += c["extra_args"].split()
    return cmd


def running():
    return _proc is not None and _proc.poll() is None


def exit_code():
    return _proc.poll() if _proc else None


def start():
    global _proc
    with _lock:
        if running():
            return "Already running"
        for k in ("llama_server", "model"):
            if not CONF[k] or not Path(CONF[k]).exists():
                return f"Missing file: {CONF[k]}"
        log = open(LOG_PATH, "w")
        log.write("$ " + " ".join(llama_cmd()) + "\n\n")
        log.flush()
        _proc = subprocess.Popen(llama_cmd(), stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        return "Starting"


def stop():
    global _proc
    with _lock:
        if running():
            os.killpg(_proc.pid, signal.SIGTERM)
            try:
                _proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(_proc.pid, signal.SIGKILL)
                _proc.wait()
        _proc = None
        return "Stopped"


def health():
    """None = not reachable, 'loading', 'ok'."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=1) as r:
            return json.loads(r.read()).get("status", "ok")
    except urllib.error.HTTPError as e:
        return "loading" if e.code == 503 else f"error {e.code}"
    except Exception:
        return None


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
        mine = 0
        if running():
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
            for line in out.splitlines():
                pid, mem = [x.strip() for x in line.split(",")]
                if pid == str(_proc.pid):
                    mine = int(mem)
        return {"name": name, "used": int(used), "total": int(total), "util": int(util), "model": mine}
    except Exception:
        return None


def log_tail(n=60):
    try:
        return "".join(LOG_PATH.read_text(errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""


# ---------------- chat (used by the app) ----------------
def chat_stream(messages, effort, on_reasoning, on_content, cancel):
    """Stream a chat completion. Calls on_reasoning/on_content with text pieces.
    Returns (timings dict or None). Raises on HTTP errors."""
    req_body = {"model": MODEL_ID, "stream": True, "messages": messages,
                "stream_options": {"include_usage": True}}
    if effort is not None:  # None = use the server default from the Model tab
        req_body["chat_template_kwargs"] = {"reasoning_effort": float(effort)}
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read())["error"]["message"]
        except Exception:
            msg = f"HTTP {e.code}"
        raise RuntimeError(msg)
    except urllib.error.URLError:
        raise RuntimeError("The model isn't running. Start it in the Model tab.")
    timings = None
    with r:
        for raw in r:
            if cancel.is_set():
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            timings = d.get("timings") or timings
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("reasoning_content"):
                    on_reasoning(delta["reasoning_content"])
                if delta.get("content"):
                    on_content(delta["content"])
    return timings


def load_chats():
    try:
        return json.loads(CHATS_PATH.read_text())
    except Exception:
        return []


def save_chats(chats):
    tmp = CHATS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(chats))
    os.chmod(tmp, 0o600)
    tmp.replace(CHATS_PATH)


# ---------------- dsh ----------------
DSH_PROVIDER_ID = "inkling-local"


def dsh_profiles():
    d = DSH_HOME / "profiles"
    return sorted(p.name for p in d.iterdir() if (p / "cordis.patch.yml").exists()) if d.exists() else []


def dsh_patch():
    return DSH_HOME / "profiles" / CONF.get("dsh_profile", "web") / "cordis.patch.yml"


def dsh_state():
    if not dsh_patch().exists():
        return {"installed": False}
    import yaml
    try:
        entries = yaml.safe_load(dsh_patch().read_text()) or []
    except Exception as e:
        return {"installed": True, "error": str(e)}
    prov, default = False, None
    for e in entries:
        if e.get("id") == "llm-pi-ai":
            prov = DSH_PROVIDER_ID in (((e.get("config") or {}).get("providers")) or {})
        if e.get("id") == "agent-default-model":
            default = e.get("config") or {}
    return {"installed": True, "provider": prov, "default": default,
            "is_default": bool(default and default.get("provider") == DSH_PROVIDER_ID)}


def dsh_connect(make_default):
    import yaml
    text = dsh_patch().read_text()
    entries = yaml.safe_load(text) or []  # raises on !!js tags -> we refuse rather than mangle
    backup = dsh_patch().with_name(f"cordis.patch.yml.before-inkling-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(dsh_patch(), backup)
    header = "".join(l for l in text.splitlines(True) if l.startswith("#"))
    provider = {
        "displayName": "Inkling 2 Mini (this PC)",
        "api": "openai-completions",
        "baseURL": f"http://127.0.0.1:{LLAMA_PORT}/v1",
        "models": [{"id": MODEL_ID, "name": "Inkling 2 Mini", "contextWindow": int(CONF["ctx"]),
                    "maxTokens": min(16384, int(CONF["ctx"]) // 2),
                    "input": ["text", "image"] if CONF["vision_mode"] != "off" else ["text"]}],
    }
    for e in entries:
        if e.get("id") == "llm-pi-ai":
            cfg = e.setdefault("config", {}) or {}
            e["config"] = cfg
            cfg["providers"] = cfg.get("providers") or {}
            cfg["providers"][DSH_PROVIDER_ID] = provider
            break
    else:
        entries.append({"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai",
                        "config": {"providers": {DSH_PROVIDER_ID: provider}}})
    if make_default:
        for e in entries:
            if e.get("id") == "agent-default-model":
                e["config"] = {"provider": DSH_PROVIDER_ID, "model": MODEL_ID}
                break
        else:
            entries.append({"id": "agent-default-model", "name": "@deepseek-ai/dsh-agent-default-model",
                            "config": {"provider": DSH_PROVIDER_ID, "model": MODEL_ID}})
    dsh_patch().write_text(header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    return backup.name


# ---------------- LAN API gateway ----------------
class _Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _handle(self):
        if not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {CONF['token']}"):
            return self._json({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}, 401)
        path = self.path.split("?")[0]
        if not path.startswith("/v1/"):
            return self._json({"error": {"message": "Not found"}}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}{path}", data=body, method=self.command,
                                     headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            r = e
        except Exception:
            return self._json({"error": {"message": "Inkling isn't running on the PC right now."}}, 503)
        stream = "text/event-stream" in r.headers.get("Content-Type", "")
        self.send_response(getattr(r, "status", None) or r.code)
        self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
        if stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while chunk := r.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
        else:
            data = r.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    do_GET = do_POST = _handle

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()


class _LanServer:
    def __init__(self):
        self.srv = None
        self.error = None

    def set(self, on):
        if on and not self.srv:
            try:
                self.srv = ThreadingHTTPServer(("0.0.0.0", LAN_PORT), _Gateway)
                self.srv.daemon_threads = True
                threading.Thread(target=self.srv.serve_forever, daemon=True).start()
                self.error = None
            except OSError as e:
                self.error = str(e)
        elif not on and self.srv:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None


lan = _LanServer()


def set_lan(on):
    CONF["lan"] = bool(on)
    save_conf()
    lan.set(CONF["lan"])


def new_token():
    CONF["token"] = "sk-inkwell-" + secrets.token_urlsafe(18)
    save_conf()
