"""First-run setup: pick a GPU backend, install build tools, choose a model and where to keep it,
then build llama.cpp and download everything with live progress."""
import shutil, subprocess, threading
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk, Pango

from . import builder, catalog, core, hardware


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return f"{n:.1f} {u}" if u in ("GB", "TB") else f"{n:.0f} {u}"
        n /= 1000


def free_space(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def page(title, child, tag):
    tv = Adw.ToolbarView()
    tv.add_top_bar(Adw.HeaderBar())
    tv.set_content(child)
    return Adw.NavigationPage(title=title, child=tv, tag=tag)


def scrolled_prefs(*groups, description=None, title=None, icon=None):
    pp = Adw.PreferencesPage()
    if title:
        head = Adw.PreferencesGroup()
        sp = Adw.StatusPage(title=title, description=description or "", icon_name=icon or "")
        sp.add_css_class("compact")
        head.add(sp)
        pp.add(head)
    for g in groups:
        pp.add(g)
    return pp


def footer_button(label, cb, sensitive=True):
    b = Gtk.Button(label=label, css_classes=["suggested-action", "pill"], halign=Gtk.Align.CENTER,
                   margin_top=12, margin_bottom=18, sensitive=sensitive)
    b.connect("clicked", lambda *_: cb())
    return b


class Wizard(Adw.ApplicationWindow):
    def __init__(self, app, on_done, rebuild_only=False, model_only=False):
        super().__init__(application=app, title="Set up Inkwell", default_width=760, default_height=720)
        self.on_done = on_done
        self.rebuild_only, self.model_only = rebuild_only, model_only
        self.sys = hardware.detect()
        self.quants, self.live = [], False
        self.choice = {
            "backend": core.CONF.get("backend") or hardware.recommend_backend(self.sys),
            "llama_mode": core.CONF.get("llama_mode", "recommended"),
            "llama_src": core.CONF["llama_src"],
            "models_dir": core.CONF["models_dir"],
            "quant": None, "vision": core.CONF.get("vision_mode", "cpu") != "off", "existing": None,
        }
        self.toasts = Adw.ToastOverlay()
        self.nav = Adw.NavigationView()
        self.toasts.set_child(self.nav)
        self.set_content(self.toasts)
        threading.Thread(target=self._load_quants, daemon=True).start()
        if rebuild_only:
            self.nav.add(self.page_backend())
        elif model_only:
            self.nav.add(self.page_model())
        else:
            self.nav.add(self.page_welcome())

    def toast(self, msg):
        self.toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(msg), timeout=4))

    def _load_quants(self):
        q, live = catalog.fetch_quants()
        GLib.idle_add(self._quants_ready, q, live)

    def _quants_ready(self, q, live):
        self.quants, self.live = q, live
        if hasattr(self, "_model_list"):
            self.fill_models()
        return False

    # ---------- 1. welcome ----------
    def page_welcome(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sp = Adw.StatusPage(icon_name="io.github.Ezmanw.Inkwell", title="Welcome to Inkwell",
                            description="Run <b>Inkling 2 Mini</b> on your own PC: chat with it, use it from "
                                        "your phone, and plug it into coding agents like dsh.\n\n"
                                        "Setup builds llama.cpp for your graphics card and downloads the model. "
                                        "Plan on 10–30 minutes and about 10–25 GB of disk space.",
                            vexpand=True)
        box.append(sp)
        box.append(footer_button("Get started", lambda: self.nav.push(self.page_backend())))
        return page("Inkwell", box, "welcome")

    # ---------- 2. hardware + backend ----------
    def page_backend(self):
        hw = Adw.PreferencesGroup(title="Your computer")
        if self.sys.gpus:
            for g in self.sys.gpus:
                sub = f"{g.vram_mb / 1024:.0f} GB video memory" if g.vram_mb else "Video memory unknown"
                if g.cuda_max:
                    sub += f" · driver supports CUDA {g.cuda_max}"
                if g.gfx:
                    sub += f" · {g.gfx}"
                hw.add(Adw.ActionRow(title=g.name, subtitle=sub, icon_name="video-display-symbolic"))
        else:
            hw.add(Adw.ActionRow(title="No graphics card found", subtitle="Inkling will run on the CPU (slow)",
                                 icon_name="dialog-warning-symbolic"))
        hw.add(Adw.ActionRow(title="System memory", subtitle=f"{self.sys.ram_mb / 1024:.0f} GB",
                             icon_name="drive-harddisk-symbolic"))

        rec = hardware.recommend_backend(self.sys)
        bg = Adw.PreferencesGroup(title="How should Inkling use your graphics card?",
                                  description="Pick one. You can rebuild with a different one later.")
        first = None
        for key, info in hardware.BACKENDS.items():
            row = Adw.ActionRow(title=info["title"] + ("  ·  Recommended" if key == rec else ""),
                                subtitle=f"{info['for']}. {info['desc']}")
            chk = Gtk.CheckButton(valign=Gtk.Align.CENTER)
            if first:
                chk.set_group(first)
            else:
                first = chk
            chk.set_active(key == self.choice["backend"])
            chk.connect("toggled", lambda c, k=key: c.get_active() and self.choice.update(backend=k))
            row.add_prefix(chk)
            row.set_activatable_widget(chk)
            bg.add(row)

        adv = Adw.PreferencesGroup(title="llama.cpp version")
        mode = Adw.ComboRow(title="Version", subtitle="Recommended is the exact version Inkwell was tested with")
        mode.set_model(Gtk.StringList.new(["Recommended: tested, with the Inkling VRAM fix",
                                           "Latest: newest llama.cpp + the fix (if still needed)"]))
        mode.set_selected(0 if self.choice["llama_mode"] == "recommended" else 1)
        mode.connect("notify::selected", lambda r, _p: self.choice.update(
            llama_mode="recommended" if r.get_selected() == 0 else "latest"))
        adv.add(mode)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pp = scrolled_prefs(hw, bg, adv)
        pp.set_vexpand(True)
        box.append(pp)
        box.append(footer_button("Next", lambda: self.nav.push(self.page_deps())))
        return page("Graphics card", box, "backend")

    # ---------- 3. dependencies ----------
    def page_deps(self):
        b = self.choice["backend"]
        g = Adw.PreferencesGroup(title="Build tools",
                                 description=f"Needed to build llama.cpp with {hardware.BACKENDS[b]['title']}.")
        self._deps_group = g
        self._deps_rows = []
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._deps_log = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.CHAR,
                                      top_margin=8, bottom_margin=8, left_margin=8, right_margin=8)
        logexp = Adw.ExpanderRow(title="Installer output")
        lsw = Gtk.ScrolledWindow(min_content_height=200)
        lsw.set_child(self._deps_log)
        logexp.add_row(lsw)
        lg = Adw.PreferencesGroup()
        lg.add(logexp)
        self._deps_logexp = logexp
        pp = scrolled_prefs(g, lg)
        pp.set_vexpand(True)
        box.append(pp)
        btns = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER, margin_top=12, margin_bottom=18)
        self._install_btn = Gtk.Button(label="Install missing tools", css_classes=["pill"])
        self._install_btn.connect("clicked", lambda *_: self.install_deps())
        self._deps_next = Gtk.Button(label="Next", css_classes=["suggested-action", "pill"])
        self._deps_next.connect("clicked", lambda *_: self.nav.push(
            self.page_install() if self.rebuild_only else self.page_storage()))
        btns.append(self._install_btn)
        btns.append(self._deps_next)
        box.append(btns)
        self.refresh_deps()
        return page("Build tools", box, "deps")

    def refresh_deps(self):
        for r in self._deps_rows:
            self._deps_group.remove(r)
        self._deps_rows = []
        miss = hardware.missing_tools(self.choice["backend"])
        if not miss:
            r = Adw.ActionRow(title="Everything's installed", icon_name="emblem-ok-symbolic")
            self._deps_group.add(r)
            self._deps_rows.append(r)
        for what, why in miss:
            r = Adw.ActionRow(title=what, subtitle=f"Missing · {why}", icon_name="dialog-warning-symbolic")
            self._deps_group.add(r)
            self._deps_rows.append(r)
        can_auto = self.sys.is_debian_like and shutil.which("pkexec") and shutil.which("apt-get")
        self._install_btn.set_visible(bool(miss) and bool(can_auto))
        self._deps_next.set_sensitive(not miss)
        if miss and not can_auto:
            r = Adw.ActionRow(title="Install these with your package manager, then reopen Inkwell",
                              subtitle="Automatic install only works on Debian/Ubuntu-based systems")
            self._deps_group.add(r)
            self._deps_rows.append(r)
        if miss and self.choice["backend"] == "cuda" and can_auto:
            r = Adw.ActionRow(title="CUDA comes from NVIDIA's package repository",
                              subtitle="Inkwell adds it at low priority so it never replaces your graphics driver. "
                                       "It's about 3 GB.")
            self._deps_group.add(r)
            self._deps_rows.append(r)

    def install_deps(self):
        self._install_btn.set_sensitive(False)
        self._install_btn.set_label("Installing…")
        self._deps_logexp.set_expanded(True)
        buf = self._deps_log.get_buffer()
        cmd = builder.deps_command(self.choice["backend"], self.sys)

        def log(t):
            GLib.idle_add(lambda: buf.insert(buf.get_end_iter(), t) or False)

        def work():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    log(line)
                rc = p.wait()
            except FileNotFoundError as e:
                log(str(e)); rc = 1
            GLib.idle_add(done, rc)

        def done(rc):
            self._install_btn.set_sensitive(True)
            self._install_btn.set_label("Install missing tools")
            if rc in (126, 127):
                self.toast("Install cancelled (password dialog was closed).")
            elif rc != 0:
                self.toast("Install failed. See the installer output.")
            self.refresh_deps()
            return False
        threading.Thread(target=work, daemon=True).start()

    # ---------- 4. storage ----------
    def page_storage(self):
        g = Adw.PreferencesGroup(title="Where should Inkwell keep things?")
        self._store_rows = {}
        for key, title, sub in [("models_dir", "Models", "The model file is 8–23 GB. Pick a folder that already "
                                                         "has Inkling files and they won't be downloaded again"),
                                ("llama_src", "llama.cpp", "Source + build, about 1–2 GB")]:
            row = Adw.ActionRow(title=title, subtitle_selectable=True)
            row.add_css_class("property")
            b = Gtk.Button(label="Change…", valign=Gtk.Align.CENTER, css_classes=["flat"])
            b.connect("clicked", lambda _b, k=key: self.pick_folder(k))
            row.add_suffix(b)
            row.hint = sub
            g.add(row)
            self._store_rows[key] = row
        self.update_storage()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pp = scrolled_prefs(g)
        pp.set_vexpand(True)
        box.append(pp)
        box.append(footer_button("Next", lambda: self.nav.push(self.page_model())))
        return page("Storage", box, "storage")

    def update_storage(self):
        for key, row in self._store_rows.items():
            path = self.choice[key]
            row.set_subtitle(f"{path}\n{row.hint} · {human(free_space(path))} free here")

    def pick_folder(self, key):
        dlg = Gtk.FileDialog(title="Choose a folder", initial_folder=Gio.File.new_for_path(
            str(Path(self.choice[key]).parent if not Path(self.choice[key]).exists() else self.choice[key])))

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            base = Path(f.get_path())
            # Reuse a folder that already holds what we need; only nest inside unrelated, non-empty folders.
            reuse = (base / ".git").exists() if key == "llama_src" else any(base.glob("*.gguf"))
            if reuse or not any(base.iterdir()):
                self.choice[key] = str(base)
            else:
                self.choice[key] = str(base / ("llama.cpp" if key == "llama_src" else "inkwell-models"))
            self.update_storage()
        dlg.select_folder(self, None, done)

    # ---------- 5. model ----------
    def page_model(self):
        g = self.sys.best_gpu
        vram = g.vram_mb if g else 0
        desc = (f"Sized for your {vram / 1024:.0f} GB graphics card and {self.sys.ram_mb / 1024:.0f} GB of RAM. "
                "Bigger files are smarter but need more memory.") if vram else \
            "No graphics card memory detected: everything runs from system RAM."
        self._model_group = Adw.PreferencesGroup(title="Pick a model size", description=desc)
        self._model_list = []
        self._model_first = None
        self.fill_models()

        opt = Adw.PreferencesGroup(title="Options")
        vis = Adw.SwitchRow(title="Image input", subtitle="Lets Inkling look at pictures · +639 MB download",
                            active=self.choice["vision"])
        vis.connect("notify::active", lambda r, _p: self.choice.update(vision=r.get_active()))
        opt.add(vis)
        have = Adw.ActionRow(title="I already have a GGUF file",
                             subtitle="Use an Inkling 2 Mini .gguf you downloaded before (mainline layout)")
        hb = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER, css_classes=["flat"])
        hb.connect("clicked", lambda *_: self.pick_existing())
        have.add_suffix(hb)
        self._existing_row = have
        opt.add(have)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pp = scrolled_prefs(self._model_group, opt)
        pp.set_vexpand(True)
        box.append(pp)
        box.append(footer_button("Set up Inkwell" if not self.model_only else "Download",
                                 lambda: self.nav.push(self.page_install())))
        return page("Model", box, "model")

    def fill_models(self):
        for r in self._model_list:
            self._model_group.remove(r)
        self._model_list = []
        if not self.quants:
            r = Adw.ActionRow(title="Loading the model list…")
            r.add_suffix(Gtk.Spinner(spinning=True))
            self._model_group.add(r)
            self._model_list.append(r)
            return
        g = self.sys.best_gpu
        vram = g.vram_mb if g else 0
        rec = catalog.recommend(self.quants, vram, self.sys.ram_mb)
        if not self.choice["quant"]:
            self.choice["quant"] = rec
        first = None
        for q in self.quants:
            fit = catalog.plan_fit(q.size, vram, self.sys.ram_mb)
            st = {"gpu": "Fits on your graphics card",
                  "split": f"Partly in system RAM ({fit['cpu_layers']} layers' experts) · a bit slower",
                  "slow": "Too big for the card. Much of it runs on the CPU, so it's slow",
                  "ram_tight": "Not enough memory on this PC",
                  "cpu": "Runs from system RAM"}[fit["status"]]
            row = Adw.ActionRow(title=f"{q.short}  ·  {q.gb:.1f} GB" + ("  ·  Recommended" if q is rec else ""),
                                subtitle=f"{q.quality} quality · {st}")
            chk = Gtk.CheckButton(valign=Gtk.Align.CENTER)
            if first:
                chk.set_group(first)
            else:
                first = chk
            chk.set_active(q is self.choice["quant"])
            chk.connect("toggled", lambda c, q=q: c.get_active() and self.choice.update(quant=q, existing=None))
            row.add_prefix(chk)
            row.set_activatable_widget(chk)
            row.set_sensitive(fit["status"] != "ram_tight")
            self._model_group.add(row)
            self._model_list.append(row)
        if not self.live:
            r = Adw.ActionRow(title="Couldn't reach Hugging Face", subtitle="Showing the built-in list; sizes may be out of date")
            self._model_group.add(r)
            self._model_list.append(r)

    def pick_existing(self):
        f = Gtk.FileFilter(name="GGUF models")
        f.add_pattern("*.gguf")
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(f)

        def done(d, res):
            try:
                path = d.open_finish(res).get_path()
            except GLib.Error:
                return
            self.choice["existing"] = path
            self._existing_row.set_subtitle(f"Using {path}")
        Gtk.FileDialog(title="Choose your Inkling 2 Mini GGUF", filters=store).open(self, None, done)

    # ---------- 6. install ----------
    def page_install(self):
        c = self.choice
        self.steps = {}
        g = Adw.PreferencesGroup(title="Setting up", description="You can leave this running. Closing the window "
                                                                 "pauses it; reopening Inkwell picks up where it left off.")
        plan = []
        if not self.model_only:
            plan.append(("build", f"Build llama.cpp ({hardware.BACKENDS[c['backend']]['title']})"))
        if not self.rebuild_only:
            if not c["existing"]:
                plan.append(("model", f"Download {c['quant'].short} model ({c['quant'].gb:.1f} GB)"))
            if c["vision"]:
                plan.append(("vision", "Download image support (0.6 GB)"))
        for key, title in plan:
            row = Adw.ActionRow(title=title, subtitle="Waiting…")
            bar = Gtk.ProgressBar(valign=Gtk.Align.CENTER, hexpand=False)
            bar.set_size_request(180, -1)
            icon = Gtk.Image(icon_name="content-loading-symbolic")
            row.add_suffix(bar)
            row.add_prefix(icon)
            g.add(row)
            self.steps[key] = (row, bar, icon)
        self._log = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.CHAR,
                                 top_margin=8, bottom_margin=8, left_margin=8, right_margin=8)
        self._log_sw = Gtk.ScrolledWindow(min_content_height=240)
        self._log_sw.set_child(self._log)
        lexp = Adw.ExpanderRow(title="Details")
        lexp.add_row(self._log_sw)
        lg = Adw.PreferencesGroup()
        lg.add(lexp)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        pp = scrolled_prefs(g, lg)
        pp.set_vexpand(True)
        box.append(pp)
        self._finish_btn = footer_button("Open Inkwell", self.finish, sensitive=False)
        self._retry_btn = Gtk.Button(label="Try again", css_classes=["pill"], halign=Gtk.Align.CENTER,
                                     margin_bottom=18, visible=False)
        self._retry_btn.connect("clicked", lambda *_: self.start_install())
        box.append(self._finish_btn)
        box.append(self._retry_btn)
        p = page("Installing", box, "install")
        p.set_can_pop(False)
        GLib.idle_add(self.start_install)
        return p

    def set_step(self, key, state=None, text=None, frac=None):
        def ui():
            row, bar, icon = self.steps[key]
            if text is not None:
                row.set_subtitle(text)
            if frac is None:
                bar.pulse()
            else:
                bar.set_fraction(max(0, min(1, frac)))
            if state:
                icon.set_from_icon_name({"run": "content-loading-symbolic", "ok": "emblem-ok-symbolic",
                                         "err": "dialog-error-symbolic"}[state])
            return False
        GLib.idle_add(ui)

    def log(self, t):
        def ui():
            buf = self._log.get_buffer()
            buf.insert(buf.get_end_iter(), t)
            if buf.get_line_count() > 4000:
                buf.delete(buf.get_start_iter(), buf.get_iter_at_line(1000)[1])
            adj = self._log_sw.get_vadjustment()
            adj.set_value(adj.get_upper())
            return False
        GLib.idle_add(ui)

    def start_install(self):
        self._retry_btn.set_visible(False)
        self._finish_btn.set_sensitive(False)
        self.runner = builder.Runner(self.log)
        self.results = {}
        c = self.choice
        threads = []

        def job(key, fn):
            def run():
                try:
                    self.set_step(key, "run", "Starting…", 0)
                    self.results[key] = fn()
                except builder.Cancelled:
                    self.results[key] = None
                    self.set_step(key, "err", "Paused", None)
                except Exception as e:
                    self.results[key] = e
                    self.log(f"\n!! {e}\n")
                    self.set_step(key, "err", str(e).splitlines()[0][:140], None)
            t = threading.Thread(target=run, daemon=True)
            threads.append(t)

        if "build" in self.steps:
            def do_build():
                st = lambda txt, f: self.set_step("build", None, txt, f)
                fix = builder.prepare_source(self.runner, c["llama_src"], c["llama_mode"], st)
                self.log(f"\n[VRAM fix: {fix}]\n")
                exe = builder.build(self.runner, c["llama_src"], c["backend"], self.sys, st,
                                    lambda f: self.set_step("build", None, f"Compiling… {f * 100:.0f}%", f))
                ok, ver = builder.binary_works(exe)
                if not ok:
                    raise RuntimeError("llama-server was built but won't run: " + " ".join(ver))
                note = {"fix-merged": "with the VRAM fix", "fix-already-included": "VRAM fix already included",
                        "no-fix": "WITHOUT the VRAM fix (it couldn't be applied)"}[fix]
                self.set_step("build", "ok", f"Done · {note}", 1)
                return exe
            job("build", do_build)

        def dl(key, url, dest, size):
            def f():
                def prog(done, total, bps):
                    txt = f"{human(done)} of {human(total)}" + (f" · {human(bps)}/s" if bps else "")
                    if bps and total > done:
                        eta = (total - done) / bps
                        txt += f" · {int(eta // 60)} min {int(eta % 60)} s left"
                    self.set_step(key, None, txt, done / total if total else None)
                path = builder.download(self.runner, url, dest, size, prog)
                self.set_step(key, "ok", f"Done · {Path(path).name}", 1)
                return path
            return f

        mdir = Path(c["models_dir"])
        if "model" in self.steps:
            q = c["quant"]
            job("model", dl("model", q.url, mdir / q.filename, q.size))
        if "vision" in self.steps:
            job("vision", dl("vision", catalog.mmproj_url(), mdir / catalog.MMPROJ[0], catalog.MMPROJ[1]))

        def watch():
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            GLib.idle_add(self.install_done)
        threading.Thread(target=watch, daemon=True).start()
        return False

    def install_done(self):
        errs = [k for k, v in self.results.items() if isinstance(v, Exception) or v is None]
        if errs:
            self._retry_btn.set_visible(True)
            self.toast("Something went wrong. Open Details to see why, then try again.")
            return False
        c, conf = self.choice, core.CONF
        conf["backend"], conf["llama_mode"], conf["llama_src"] = c["backend"], c["llama_mode"], c["llama_src"]
        conf["models_dir"] = c["models_dir"]
        if "build" in self.results:
            conf["llama_server"] = self.results["build"]
        if not self.rebuild_only:
            conf["model"] = c["existing"] or self.results.get("model") or conf["model"]
            if c["vision"]:
                conf["mmproj"] = self.results.get("vision") or conf["mmproj"]
                conf["vision_mode"] = conf["vision_mode"] if conf["vision_mode"] != "off" else "cpu"
            else:
                conf["vision_mode"] = "off"
            g = self.sys.best_gpu
            size = Path(conf["model"]).stat().st_size if conf["model"] else 0
            fit = catalog.plan_fit(size, g.vram_mb if g else 0, self.sys.ram_mb, conf["vision_mode"] == "gpu")
            conf["cpu_layers"] = fit["cpu_layers"] if fit["status"] in ("split", "ram_tight") else 0
            conf["gpu_layers"] = 0 if c["backend"] == "cpu" else fit["gpu_layers"]
        conf["setup_done"] = True
        core.save_conf()
        self._finish_btn.set_sensitive(True)
        return False

    def finish(self):
        self.on_done()
        self.close()

    def do_close_request(self):
        if getattr(self, "runner", None):
            self.runner.stop()
        return False
