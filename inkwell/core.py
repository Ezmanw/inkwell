"""Inkwell backend: runs llama-server, exposes a key-protected OpenAI-compatible
API on the LAN (for phone apps), and connects dsh.

Ports:
  8686  llama-server (this PC only, no key)  -> dsh / local apps
  8691  LAN API gateway (only while Wi-Fi sharing is on; Bearer key required)
"""
import json, os, re, secrets, shlex, shutil, signal, socket, subprocess, threading, time
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
    "manual": False,          # Manual mode: run llama-server with exactly manual_args, nothing added
    "manual_args": "",
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
    if CONF.get("manual"):
        return [CONF["llama_server"]] + manual_argv()
    return auto_cmd()


def manual_argv(text=None):
    text = CONF.get("manual_args", "") if text is None else text
    return shlex.split(text, comments=True)   # newlines are just spaces; '#' starts a comment


# Characters that sneak in when flags are copied from web pages or chats.
_INVISIBLE = {"\u00a0": "non-breaking space", "\u2007": "figure space", "\u202f": "narrow no-break space",
              "\u200b": "zero-width space", "\u200c": "zero-width non-joiner", "\u200d": "zero-width joiner",
              "\u2060": "word joiner", "\ufeff": "invisible BOM", "\u00ad": "soft hyphen"}
_LOOKALIKE = {"\u2013": ("en dash", "-"), "\u2014": ("em dash", "--"), "\u2212": ("minus sign", "-"),
              "\u201c": ("curly quote", '"'), "\u201d": ("curly quote", '"'),
              "\u2018": ("curly quote", "'"), "\u2019": ("curly quote", "'")}


def manual_issues(text):
    """Problems a human can't see: [(line number, message)]."""
    import unicodedata
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0] if "#" in line else line
        for ch in sorted(set(code)):
            if ch in _INVISIBLE:
                out.append((n, f"hidden {_INVISIBLE[ch]}"))
            elif ch in _LOOKALIKE:
                out.append((n, f"{_LOOKALIKE[ch][0]} (looks like {_LOOKALIKE[ch][1]!r} but isn't)"))
            elif ch not in "\t" and unicodedata.category(ch) in ("Cc", "Cf", "Zs") and ch != " ":
                out.append((n, f"invisible character U+{ord(ch):04X}"))
        try:
            toks = shlex.split(code, comments=False)
        except ValueError:
            continue
        if "" in toks:
            out.append((n, "empty argument (\"\" or '')"))
    return list(dict.fromkeys(out))


def manual_clean(text):
    """Fix what's safe to fix: hidden spaces -> spaces, zero-width chars removed, curly quotes and
    dashes -> plain ones. Empty arguments are left for the user to look at."""
    for ch in _INVISIBLE:
        text = text.replace(ch, " " if ch in "\u00a0\u2007\u202f" else "")
    for ch, (_, rep) in _LOOKALIKE.items():
        text = text.replace(ch, rep)
    return text


def manual_default_text():
    """Inkwell's own command, one flag per line, as a starting point for Manual mode."""
    args, lines, cur = auto_cmd()[1:], [], []
    for a in args:
        if a.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", a) and cur:
            lines.append(" ".join(shlex.quote(x) for x in cur)); cur = []
        cur.append(a)
    if cur:
        lines.append(" ".join(shlex.quote(x) for x in cur))
    return "\n".join(lines) + "\n"


def llama_addr():
    """(host, port) to reach llama-server. Manual mode follows the user's own --host/--port."""
    if not CONF.get("manual"):
        return "127.0.0.1", LLAMA_PORT
    host, port = "127.0.0.1", 8080          # llama-server's built-in defaults
    try:
        argv = manual_argv()
    except ValueError:
        argv = []
    for i, a in enumerate(argv):
        k, _, v = a.partition("=")
        if k == "--port":
            v = v or (argv[i + 1] if i + 1 < len(argv) else "")
            port = int(v) if v.isdigit() else port
        elif k == "--host":
            v = v or (argv[i + 1] if i + 1 < len(argv) else "")
            host = v if v and v not in ("0.0.0.0", "::", "[::]") else host
    return host, port


def llama_url(path=""):
    h, p = llama_addr()
    return f"http://{h}:{p}{path}"


def auto_cmd():
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
        try:
            cmd += shlex.split(c["extra_args"])  # quotes work: --flag "value with spaces"
        except ValueError:
            cmd += c["extra_args"].split()
    return cmd


ENGINE_NAMES = {"cuda": "CUDA", "vulkan": "Vulkan", "rocm": "ROCm", "cpu": "CPU only"}


def engines():
    """Every llama-server build we know of: {backend: path}. Builds live side by side in
    <llama_src>/build-<backend>, so switching between them is just a restart."""
    out = {}
    src = Path(CONF.get("llama_src") or "")
    for exe in sorted(src.glob("build-*/bin/llama-server")):
        out[exe.parent.parent.name[len("build-"):]] = str(exe)
    for b, pth in (CONF.get("extra_engines") or {}).items():
        if Path(pth).exists():
            out.setdefault(b, pth)
    cur = CONF.get("llama_server")
    if cur and Path(cur).exists() and cur not in out.values():
        out[CONF.get("backend") or "custom"] = cur
    return out


def remember_current_engine():
    """Keep an out-of-tree build (e.g. one you compiled yourself) in the Engine list after switching away."""
    old = CONF.get("llama_server")
    in_tree = [str(p) for p in Path(CONF.get("llama_src") or "/nonexistent").glob("build-*/bin/llama-server")]
    if old and Path(old).exists() and old not in in_tree:
        CONF.setdefault("extra_engines", {})[CONF.get("backend") or "custom"] = old


def switch_engine(backend):
    path = engines()[backend]
    remember_current_engine()
    CONF["backend"], CONF["llama_server"] = backend, path
    CONF["gpu_layers"] = 0 if backend == "cpu" else (CONF["gpu_layers"] or 99)
    save_conf()
    stop()
    return start()


def running():
    return _proc is not None and _proc.poll() is None


def exit_code():
    return _proc.poll() if _proc else None


def start():
    global _proc
    with _lock:
        if running():
            return "Already running"
        for k in (("llama_server",) if CONF.get("manual") else ("llama_server", "model")):
            if not CONF[k] or not Path(CONF[k]).exists():
                return f"Missing file: {CONF[k]}"
        log = open(LOG_PATH, "w")
        log.write("$ " + " ".join(llama_cmd()) + "\n\n")
        log.flush()
        activity.started_at = time.time()
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
        with urllib.request.urlopen(llama_url("/health"), timeout=1) as r:
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


# ---------------- activity (parsed from llama-server's log, so it covers every app) ----------------
class Activity:
    """Turns llama-server log lines into live status + a plain-English event feed."""
    RX = {
        "start": re.compile(r"launch_slot_.*\| task (\d+) \| processing task"),
        "prompt": re.compile(r"\| task (\d+) \| prompt processing, n_tokens =\s*(\d+), progress = ([\d.]+), "
                             r"t =\s*([\d.]+) s / ([\d.]+) tokens per second"),
        "gen": re.compile(r"\| task (\d+) \| n_gen =\s*(\d+), tg =\s*([\d.]+) t/s, tg_3s =\s*([\d.]+)"),
        "p_eval": re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second"),
        "eval": re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second"),
        "draft": re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)"),
        "release": re.compile(r"release: .*stop processing: n_tokens = (\d+), truncated = (\d)"),
        "cancel": re.compile(r"srv\s+stop: cancel task"),
        "error": re.compile(r"^\S+ E (.*)"),
        "loading": re.compile(r"load_model: loading model '([^']+)'"),
        "ready": re.compile(r"server_run_i: model loaded"),
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.events = []          # newest last: (time.time(), kind, text)
        self.offset = 0
        self.ino = None
        self.pending = []         # (time, source) noted by our chat / the LAN gateway
        self.reset_live()
        self.last = None          # summary of the last finished request
        self.started_at = None    # wall-clock time llama-server was launched (log times are relative to it)
        self.t = None

    def reset_live(self):
        self.state, self.task, self.source = "idle", None, ""
        self.prompt_frac = self.prompt_n = self.prompt_tps = 0
        self.gen_n = self.gen_tps = 0
        self.p_eval = None

    def note_request(self, source):
        with self.lock:
            self.pending.append((time.time(), source))

    def add(self, kind, text):
        self.events.append((self.t or time.time(), kind, text))
        del self.events[:-300]

    def poll(self):
        """Read new log lines. Cheap; call every second or two."""
        try:
            st = LOG_PATH.stat()
        except OSError:
            return
        with self.lock:
            if st.st_ino != self.ino or st.st_size < self.offset:  # server restarted -> new log
                self.ino, self.offset = st.st_ino, 0
                self.reset_live()
            with open(LOG_PATH, "r", errors="replace") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
            for line in data.splitlines():
                m = re.match(r"(\d+)\.(\d+)\.(\d+)\.\d+ ", line)  # minutes.seconds.ms.us since launch
                self.t = (self.started_at + int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
                          if m and self.started_at else None)
                self._line(line)
            self.t = None

    def _who(self):
        now = time.time()
        self.pending = [(t, s) for t, s in self.pending if now - t < 15]
        return self.pending.pop(0)[1] if self.pending else "An app on this PC (dsh or similar)"

    def _line(self, line):
        R = self.RX
        if (m := R["start"].search(line)):
            self.reset_live()
            self.state, self.task, self.source = "reading", m.group(1), self._who()
            self.add("start", f"{self.source} sent a request")
        elif (m := R["prompt"].search(line)):
            self.state = "reading"
            self.prompt_n, self.prompt_frac, self.prompt_tps = int(m.group(2)), float(m.group(3)), float(m.group(5))
        elif (m := R["gen"].search(line)):
            self.state = "writing"
            self.gen_n, self.gen_tps = int(m.group(2)), float(m.group(4))
        elif (m := R["p_eval"].search(line)):
            self.p_eval = (int(m.group(2)), float(m.group(3)))
        elif (m := R["eval"].search(line)):
            n, tps, secs = int(m.group(2)), float(m.group(3)), float(m.group(1)) / 1000
            pn, ptps = self.p_eval or (0, 0)
            self.last = {"tokens": n, "tps": tps, "secs": secs, "prompt": pn, "prompt_tps": ptps,
                         "source": self.source, "time": time.time()}
            self.add("done", f"Wrote {n:,} tokens at {tps:.1f} tok/s in {secs:.0f}s"
                             + (f" · read {pn:,} new prompt tokens at {ptps:.0f}/s" if pn > 1 else ""))
        elif (m := R["draft"].search(line)):
            if self.events and self.events[-1][1] == "done":
                t, k, txt = self.events[-1]
                self.events[-1] = (t, k, txt + f" · draft model guessed {float(m.group(1)) * 100:.0f}% right")
        elif (m := R["release"].search(line)):
            if m.group(2) == "1":
                self.add("warn", f"Context full: the conversation hit the {int(m.group(1)):,}-token limit and the "
                                 "oldest part was cut off. Raise Context size or start a new chat.")
            self.state = "idle"
        elif R["cancel"].search(line):
            self.add("stop", "The app stopped the request")
            self.state = "idle"
        elif (m := R["error"].search(line)):
            msg = re.sub(r"^\S+\s+\S+:\s*", "", m.group(1)).strip()
            if "exceeds the available context" in msg:
                msg = "A request was too long for the context size and was rejected. Raise Context size."
            self.add("error", msg[:240])
        elif (m := R["loading"].search(line)):
            self.add("info", f"Loading {Path(m.group(1)).name}…")
        elif R["ready"].search(line):
            self.add("ok", "Model loaded and ready")

    def live(self):
        with self.lock:
            return {"state": self.state, "source": self.source, "prompt_frac": self.prompt_frac,
                    "prompt_n": self.prompt_n, "prompt_tps": self.prompt_tps, "gen_n": self.gen_n,
                    "gen_tps": self.gen_tps, "last": self.last}

    def feed(self, n=150):
        with self.lock:
            return list(self.events[-n:])


activity = Activity()


def log_tail(n=60):
    try:
        return "".join(LOG_PATH.read_text(errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""


# ---------------- chat (used by the app) ----------------
def chat_stream(messages, effort, on_reasoning, on_content, cancel):
    activity.note_request("Inkwell chat")
    """Stream a chat completion. Calls on_reasoning/on_content with text pieces.
    Returns (timings dict or None). Raises on HTTP errors."""
    req_body = {"model": MODEL_ID, "stream": True, "messages": messages,
                "stream_options": {"include_usage": True}}
    if effort is not None:  # None = use the server default from the Model tab
        req_body["chat_template_kwargs"] = {"reasoning_effort": float(effort)}
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(llama_url("/v1/chat/completions"), data=body,
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
        "baseURL": llama_url("/v1"),
        # llama-server needs no key, but dsh's OpenAI client refuses to send without one
        # ("No API key for provider"); its docs say to give keyless local servers a placeholder header.
        "headers": {"Authorization": "Bearer inkwell-local"},
        "models": [{"id": MODEL_ID, "name": "Inkling 2 Mini", "contextWindow": int(CONF["ctx"]),
                    "maxTokens": min(16384, int(CONF["ctx"]) // 2),
                    "input": ["text", "image"] if CONF["vision_mode"] != "off" else ["text"],
                    # dsh level -> reasoning_effort sent to llama-server. Inkling's template knows
                    # minimal .1 / low .2 / medium .7 / high .9 / xhigh .99. There's no real "off":
                    # 'none' is buggy in llama.cpp (#32753, gives 0.9), so off -> minimal.
                    "reasoningEfforts": {"off": "minimal", "minimal": "minimal", "low": "low",
                                         "medium": "medium", "high": "high", "xhigh": "xhigh"},
                    "compat": {"supportsReasoningEffort": True}}],
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
            activity.add("warn", f"Blocked a Wi-Fi request with a wrong API key from {self.client_address[0]}")
            return self._json({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}, 401)
        path = self.path.split("?")[0]
        if self.command == "POST" and path.endswith("/completions"):
            activity.note_request(f"A device on Wi-Fi ({self.client_address[0]})")
        if not path.startswith("/v1/"):
            return self._json({"error": {"message": "Not found"}}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(llama_url(path), data=body, method=self.command,
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
