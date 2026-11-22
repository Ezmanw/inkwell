#!/usr/bin/env python3
"""Inkwell: a native GTK4/libadwaita app to run Inkling 2 Mini locally,
chat with it (text + images), share it with your phone as an API, and hook up dsh."""
import base64, re, sys, threading, time
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GdkPixbuf, Gio, GLib, Gtk, Pango

from . import core, wizard

APP_ID = "io.github.Ezmanw.Inkwell"

CSS = """
.bubble-user { background: alpha(@accent_bg_color, 0.18); border-radius: 18px 18px 4px 18px; padding: 10px 14px; }
.bubble-bot  { background: @card_bg_color; border-radius: 18px 18px 18px 4px; padding: 10px 14px; }
.codeblock   { background: alpha(@window_fg_color, 0.07); border-radius: 8px; padding: 8px 10px; font-family: monospace; }
.composer    { background: @card_bg_color; border-radius: 24px; padding: 6px 8px 6px 14px; }
.composer textview, .composer textview text { background: transparent; }
.status-pill { border-radius: 99px; padding: 3px 12px; font-size: 0.9em; background: alpha(@window_fg_color, 0.08); }
.status-ok   { color: @success_color; }
.status-warn { color: @warning_color; }
.status-err  { color: @error_color; }
.thumb       { border-radius: 8px; }
.bigstat     { font-size: 1.6em; font-weight: 600; }
.log         { font-family: monospace; font-size: 0.85em; }
"""

CTX_OPTS = [(8192, "8K: fastest, least memory"), (16384, "16K"), (32768, "32K: good for dsh"),
            (49152, "48K"), (65536, "64K"), (98304, "96K"), (131072, "128K")]
UB_OPTS = [(64, "64: least VRAM"), (128, "128"), (256, "256: balanced"), (512, "512: fastest prompts")]
KV_OPTS = [("q8_0", "q8_0: half the memory (recommended)"), ("f16", "f16: full quality"), ("q4_0", "q4_0: smallest")]
VISION_OPTS = [("cpu", "CPU: saves ~0.6 GB VRAM, images read slower"), ("gpu", "GPU: fastest, uses ~0.6 GB VRAM"),
               ("off", "Off: no images")]
# Inkling's chat template maps names to a 0..0.99 effort level; we send the number directly.
EFFORT_MARKS = [(0.0, "Off"), (0.1, None), (0.2, "Low"), (0.7, "Medium"), (0.9, "High"), (0.99, None)]
CHAT_EFFORTS = [(None, "Default"), (0.0, "Off"), (0.1, "Minimal"), (0.2, "Low"), (0.7, "Medium"),
                (0.9, "High"), (0.99, None)]
BUDGET_OPTS = [(-1, "No limit"), (256, "256 tokens: quick"), (512, "512 tokens"), (1024, "1,024 tokens"),
               (2048, "2,048 tokens"), (4096, "4,096 tokens"), (8192, "8,192 tokens")]


def effort_name(v):
    return min(CHAT_EFFORTS[1:], key=lambda e: abs(e[0] - v))[1]


# ---------------- small helpers ----------------
def md_to_pango(text):
    t = GLib.markup_escape_text(text)
    t = re.sub(r"`([^`\n]+)`", r"<tt>\1</tt>", t)
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<i>\1</i>", t)
    t = re.sub(r"^#{1,6} (.*)$", r"<b><big>\1</big></b>", t, flags=re.M)
    t = re.sub(r"^(\s*)[-*] ", r"\1• ", t, flags=re.M)
    return t


def make_label(text, markup=False, css=None):
    lbl = Gtk.Label(xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, selectable=True)
    lbl.set_can_focus(False)
    if markup:
        try:
            Pango.parse_markup(text, -1, "\0")
            lbl.set_markup(text)
        except GLib.Error:
            lbl.set_text(text)
    else:
        lbl.set_text(text)
    if css:
        lbl.add_css_class(css)
    return lbl


def image_to_data_url(path, max_side=1280):
    pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, max_side, max_side, True)
    ok, buf = pb.save_to_bufferv("jpeg", ["quality"], ["88"])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def data_url_texture(url):
    raw = base64.b64decode(url.split(",", 1)[1])
    return Gdk.Texture.new_from_bytes(GLib.Bytes.new(raw))


def copy_text(widget, text):
    widget.get_clipboard().set(text)


class QRCodeArea(Gtk.DrawingArea):
    def __init__(self, size=150):
        super().__init__(content_width=size, content_height=size)
        self.matrix = None
        self.set_draw_func(self._draw)

    def set_text(self, text):
        import qrcode
        q = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
        q.add_data(text)
        q.make(fit=True)
        self.matrix = q.get_matrix()
        self.queue_draw()

    def _draw(self, area, cr, w, h):
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        if not self.matrix:
            return
        n = len(self.matrix)
        s = min(w, h) / n
        cr.set_source_rgb(0, 0, 0)
        for y, row in enumerate(self.matrix):
            for x, v in enumerate(row):
                if v:
                    cr.rectangle(x * s, y * s, s + 0.5, s + 0.5)
        cr.fill()


# ---------------- chat page ----------------
class MessageWidget(Gtk.Box):
    def __init__(self, msg):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.msg = msg
        user = msg["role"] == "user"
        self.set_halign(Gtk.Align.END if user else Gtk.Align.START)
        self.set_margin_start(60 if user else 0)
        self.set_margin_end(0 if user else 60)
        self.add_css_class("bubble-user" if user else "bubble-bot")
        if user:
            if msg.get("images"):
                row = Gtk.Box(spacing=6, halign=Gtk.Align.END)
                for url in msg["images"]:
                    pic = Gtk.Picture.new_for_paintable(data_url_texture(url))
                    pic.set_size_request(160, 120)
                    pic.set_content_fit(Gtk.ContentFit.COVER)
                    pic.add_css_class("thumb")
                    row.append(pic)
                self.append(row)
            if msg["content"]:
                self.append(make_label(msg["content"]))
        else:
            self.think = Gtk.Expander(label="Thinking…")
            self.think_lbl = make_label("", css="dim-label")
            self.think.set_child(self.think_lbl)
            self.append(self.think)
            self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.append(self.body)
            self.meta = Gtk.Label(xalign=0, css_classes=["dim-label", "caption"])
            self.append(self.meta)
            self.refresh(final=True)

    def refresh(self, final=False):
        m = self.msg
        self.think.set_visible(bool(m.get("reasoning")))
        self.think_lbl.set_text(m.get("reasoning", ""))
        if m.get("reasoning") and not m.get("content") and not final:
            self.think.set_expanded(True)
        elif m.get("content") and self.think.get_label() == "Thinking…":
            self.think.set_label("Thought process")
            self.think.set_expanded(False)
        if final and m.get("reasoning"):
            self.think.set_label("Thought process")
        while (c := self.body.get_first_child()):
            self.body.remove(c)
        parts = re.split(r"```[\w+-]*\n?(.*?)(?:```|$)", m.get("content", ""), flags=re.S)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    self.body.append(make_label(md_to_pango(part.strip("\n")), markup=True))
            else:
                code = make_label(part.rstrip("\n"), css="codeblock")
                code.set_wrap(False)
                sw = Gtk.ScrolledWindow(vscrollbar_policy=Gtk.PolicyType.NEVER, propagate_natural_height=True)
                sw.set_child(code)
                self.body.append(sw)
        if not m.get("content") and not m.get("reasoning") and not final:
            self.body.append(Gtk.Spinner(spinning=True, halign=Gtk.Align.START))
        self.meta.set_text(m.get("meta", ""))
        self.meta.set_visible(bool(m.get("meta")))


class ChatPage(Gtk.Box):
    def __init__(self, win):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self.chats = core.load_chats()
        self.cur = None
        self.pending_images = []
        self.cancel = None

        split = Adw.OverlaySplitView(sidebar_width_fraction=0.24, min_sidebar_width=200, max_sidebar_width=280)
        self.split = split
        self.append(split)
        split.set_vexpand(True)

        # sidebar
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        side.set_margin_top(8); side.set_margin_start(8); side.set_margin_end(8); side.set_margin_bottom(8)
        newbtn = Gtk.Button(label="New chat", css_classes=["suggested-action", "pill"])
        newbtn.connect("clicked", lambda *_: self.new_chat())
        side.append(newbtn)
        self.list = Gtk.ListBox(css_classes=["navigation-sidebar"])
        self.list.connect("row-selected", self.on_row)
        sw = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        sw.set_child(self.list)
        side.append(sw)
        split.set_sidebar(side)

        # main
        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        top = Gtk.Box(spacing=6)
        top.set_margin_start(8); top.set_margin_end(8); top.set_margin_top(6)
        tog = Gtk.ToggleButton(icon_name="sidebar-show-symbolic", tooltip_text="Chats", css_classes=["flat"])
        tog.bind_property("active", split, "show-sidebar", 2 | 1)  # BIDIRECTIONAL|SYNC_CREATE
        top.append(tog)
        self.title = Gtk.Label(hexpand=True, xalign=0, ellipsize=Pango.EllipsizeMode.END, css_classes=["heading"])
        top.append(self.title)
        top.append(Gtk.Label(label="Thinking", css_classes=["dim-label"]))
        self.effort = Gtk.DropDown.new_from_strings([e[1] for e in CHAT_EFFORTS])
        self.effort.set_selected(0)
        self.effort.set_tooltip_text("Thinking effort for this chat. Default = the setting in the Model tab")
        top.append(self.effort)
        delbtn = Gtk.Button(icon_name="user-trash-symbolic", tooltip_text="Delete this chat", css_classes=["flat"])
        delbtn.connect("clicked", lambda *_: self.delete_chat())
        top.append(delbtn)
        main.append(top)

        self.scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        clamp = Adw.Clamp(maximum_size=860)
        self.msgs = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.msgs.set_margin_top(12); self.msgs.set_margin_bottom(12)
        self.msgs.set_margin_start(12); self.msgs.set_margin_end(12)
        clamp.set_child(self.msgs)
        self.scroll.set_child(clamp)
        main.append(self.scroll)

        # composer
        cclamp = Adw.Clamp(maximum_size=860)
        cbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        cbox.set_margin_start(12); cbox.set_margin_end(12); cbox.set_margin_bottom(12)
        self.thumbs = Gtk.Box(spacing=6)
        cbox.append(self.thumbs)
        comp = Gtk.Box(spacing=6, css_classes=["composer"])
        attach = Gtk.Button(icon_name="mail-attachment-symbolic", tooltip_text="Attach image",
                            css_classes=["flat", "circular"], valign=Gtk.Align.END)
        attach.connect("clicked", self.pick_image)
        self.attach_btn = attach
        comp.append(attach)
        self.input = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, hexpand=True, accepts_tab=False,
                                  top_margin=8, bottom_margin=8)
        isw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, propagate_natural_height=True,
                                 max_content_height=180, hexpand=True)
        isw.set_child(self.input)
        comp.append(isw)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.on_key)
        self.input.add_controller(keys)
        self.send_btn = Gtk.Button(icon_name="go-up-symbolic", tooltip_text="Send (Enter)",
                                   css_classes=["suggested-action", "circular"], valign=Gtk.Align.END)
        self.send_btn.connect("clicked", lambda *_: self.send())
        comp.append(self.send_btn)
        cbox.append(comp)
        cclamp.set_child(cbox)
        main.append(cclamp)

        # drag & drop images
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self.on_drop)
        main.add_controller(drop)
        split.set_content(main)

        if not self.chats:
            self.new_chat()
        else:
            self.rebuild_list(self.chats[0]["id"])

    # ----- chats -----
    def chat(self):
        return next((c for c in self.chats if c["id"] == self.cur), None)

    def new_chat(self):
        c = {"id": str(int(time.time() * 1000)), "title": "New chat", "msgs": []}
        self.chats.insert(0, c)
        core.save_chats(self.chats)
        self.rebuild_list(c["id"])
        self.input.grab_focus()

    def delete_chat(self):
        if self.cancel:
            return
        self.chats = [c for c in self.chats if c["id"] != self.cur]
        core.save_chats(self.chats)
        if not self.chats:
            return self.new_chat()
        self.rebuild_list(self.chats[0]["id"])

    def rebuild_list(self, select_id):
        self._rebuilding = True
        while (r := self.list.get_first_child()):
            self.list.remove(r)
        sel = None
        for c in self.chats:
            row = Gtk.ListBoxRow()
            row.chat_id = c["id"]
            row.set_child(Gtk.Label(label=c["title"], xalign=0, ellipsize=Pango.EllipsizeMode.END,
                                    margin_top=6, margin_bottom=6, margin_start=6))
            self.list.append(row)
            if c["id"] == select_id:
                sel = row
        self._rebuilding = False
        self.list.select_row(sel)
        self.show_chat(select_id)

    def on_row(self, lb, row):
        if row and not getattr(self, "_rebuilding", False) and not self.cancel:
            self.show_chat(row.chat_id)

    def show_chat(self, cid):
        self.cur = cid
        c = self.chat()
        self.title.set_text(c["title"] if c else "")
        while (w := self.msgs.get_first_child()):
            self.msgs.remove(w)
        if c and c["msgs"]:
            for m in c["msgs"]:
                self.msgs.append(MessageWidget(m))
        else:
            st = Adw.StatusPage(icon_name=APP_ID, title="Ask Inkling anything",
                                description="Runs entirely on your PC. Attach or drop images too.", vexpand=True)
            self.msgs.append(st)
        self.scroll_bottom()

    def scroll_bottom(self):
        def go():
            adj = self.scroll.get_vadjustment()
            adj.set_value(adj.get_upper())
            return False
        GLib.timeout_add(30, go)

    # ----- images -----
    def pick_image(self, *_):
        f = Gtk.FileFilter(name="Images")
        f.add_mime_type("image/*")
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(f)
        dlg = Gtk.FileDialog(title="Attach images", filters=store)
        dlg.open_multiple(self.win, None, self._picked)

    def _picked(self, dlg, res):
        try:
            files = dlg.open_multiple_finish(res)
        except GLib.Error:
            return
        for i in range(files.get_n_items()):
            self.add_image(files.get_item(i).get_path())

    def on_drop(self, target, value, x, y):
        for f in value.get_files():
            if f.get_path():
                self.add_image(f.get_path())
        return True

    def add_image(self, path):
        if core.CONF["vision_mode"] == "off":
            self.win.toast("Image input is off. Turn it on in the Model tab.")
            return
        try:
            url = image_to_data_url(path)
        except GLib.Error:
            self.win.toast("That file isn't an image I can read.")
            return
        self.pending_images.append(url)
        self.draw_thumbs()

    def draw_thumbs(self):
        while (w := self.thumbs.get_first_child()):
            self.thumbs.remove(w)
        for i, url in enumerate(self.pending_images):
            ov = Gtk.Overlay()
            pic = Gtk.Picture.new_for_paintable(data_url_texture(url))
            pic.set_size_request(72, 72)
            pic.set_content_fit(Gtk.ContentFit.COVER)
            pic.add_css_class("thumb")
            ov.set_child(pic)
            x = Gtk.Button(icon_name="window-close-symbolic", css_classes=["circular", "osd"],
                           halign=Gtk.Align.END, valign=Gtk.Align.START)
            x.connect("clicked", lambda _b, i=i: (self.pending_images.pop(i), self.draw_thumbs()))
            ov.add_overlay(x)
            self.thumbs.append(ov)

    # ----- sending -----
    def on_key(self, ctrl, keyval, keycode, state):
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and not (state & Gdk.ModifierType.SHIFT_MASK):
            self.send()
            return True
        return False

    def send(self):
        if self.cancel:
            self.cancel.set()
            return
        buf = self.input.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not text and not self.pending_images:
            return
        c = self.chat()
        if not c["msgs"]:
            while (w := self.msgs.get_first_child()):
                self.msgs.remove(w)
        user = {"role": "user", "content": text, "images": self.pending_images}
        c["msgs"].append(user)
        if c["title"] == "New chat":
            c["title"] = (text or "Image")[:40]
            self.title.set_text(c["title"])
            row = self.list.get_selected_row()
            if row:
                row.get_child().set_text(c["title"])
        buf.set_text("")
        self.pending_images = []
        self.draw_thumbs()
        self.msgs.append(MessageWidget(user))
        bot = {"role": "assistant", "content": "", "reasoning": ""}
        c["msgs"].append(bot)
        bw = MessageWidget(bot)
        bw.refresh()
        self.msgs.append(bw)
        self.scroll_bottom()
        core.save_chats(self.chats)

        api_msgs = []
        for m in c["msgs"][:-1]:
            if m.get("images"):
                parts = [{"type": "image_url", "image_url": {"url": u}} for u in m["images"]]
                if m["content"]:
                    parts.append({"type": "text", "text": m["content"]})
                api_msgs.append({"role": m["role"], "content": parts})
            else:
                api_msgs.append({"role": m["role"], "content": m["content"]})

        self.cancel = threading.Event()
        self.send_btn.set_icon_name("media-playback-stop-symbolic")
        self.send_btn.set_tooltip_text("Stop")
        effort = CHAT_EFFORTS[self.effort.get_selected()][0]
        state = {"dirty": False}

        def mark():
            if not state["dirty"]:
                state["dirty"] = True
                GLib.timeout_add(60, flush)

        def flush():
            state["dirty"] = False
            adj = self.scroll.get_vadjustment()
            at_end = adj.get_value() + adj.get_page_size() >= adj.get_upper() - 40
            bw.refresh()
            if at_end:
                self.scroll_bottom()
            return False

        def on_r(t):
            bot["reasoning"] += t
            GLib.idle_add(mark)

        def on_c(t):
            bot["content"] += t
            GLib.idle_add(mark)

        def work():
            t0 = time.time()
            try:
                tm = core.chat_stream(api_msgs, effort, on_r, on_c, self.cancel)
                if self.cancel.is_set():
                    bot["meta"] = "Stopped"
                elif tm:
                    bot["meta"] = (f"{tm.get('predicted_n', 0)} tokens · {tm.get('predicted_per_second', 0):.1f} tok/s"
                                   f" · prompt {tm.get('prompt_per_second', 0):.0f} tok/s")
                else:
                    bot["meta"] = f"{time.time() - t0:.1f}s"
            except Exception as e:
                bot["meta"] = f"Error: {e}"
            GLib.idle_add(done)

        def done():
            self.cancel = None
            self.send_btn.set_icon_name("go-up-symbolic")
            self.send_btn.set_tooltip_text("Send (Enter)")
            bw.refresh(final=True)
            core.save_chats(self.chats)
            self.win.last_speed = bot.get("meta", "")
            return False

        threading.Thread(target=work, daemon=True).start()


# ---------------- model page ----------------
def combo_row(title, subtitle, opts, current):
    row = Adw.ComboRow(title=title, subtitle=subtitle)
    row.set_model(Gtk.StringList.new([o[1] for o in opts]))
    keys = [o[0] for o in opts]
    row.set_selected(keys.index(current) if current in keys else 0)
    row.keys = keys
    return row


class ModelPage(Adw.PreferencesPage):
    def __init__(self, win):
        super().__init__()
        self.win = win
        c = core.CONF

        g = Adw.PreferencesGroup(title="Inkling 2 Mini", description=Path(c["model"]).name)
        self.head = g
        self.add(g)
        self.status_row = Adw.ActionRow(title="Status")
        self.start_btn = Gtk.Button(label="Start", css_classes=["suggested-action", "pill"], valign=Gtk.Align.CENTER)
        self.start_btn.connect("clicked", lambda *_: self.win.toast(core.start()))
        self.stop_btn = Gtk.Button(label="Stop", css_classes=["pill"], valign=Gtk.Align.CENTER)
        self.stop_btn.connect("clicked", lambda *_: threading.Thread(target=core.stop, daemon=True).start())
        self.status_row.add_suffix(self.start_btn)
        self.status_row.add_suffix(self.stop_btn)
        g.add(self.status_row)

        self.vram_row = Adw.ActionRow(title="Video memory")
        vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER, spacing=4)
        self.vram_lbl = Gtk.Label(xalign=1, css_classes=["numeric"])
        self.vram_bar = Gtk.LevelBar(min_value=0, max_value=1, hexpand=True)
        self.vram_bar.set_size_request(200, -1)
        self.vram_bar.add_offset_value("low", 0.80)
        self.vram_bar.add_offset_value("high", 0.93)
        self.vram_bar.add_offset_value("full", 1.0)
        vb.append(self.vram_lbl)
        vb.append(self.vram_bar)
        self.vram_row.add_suffix(vb)
        g.add(self.vram_row)
        self.speed_row = Adw.ActionRow(title="Speed", subtitle="Shows up after the first request from any app")
        g.add(self.speed_row)

        mg = Adw.PreferencesGroup(title="Manual mode")
        self.add(mg)
        self.manual = Adw.SwitchRow(title="Run llama-server with my own flags",
                                    subtitle="Inkwell adds nothing: no thinking level, sampling, context or batch "
                                             "settings. Apps like dsh set thinking per request",
                                    active=bool(c.get("manual")))
        mg.add(self.manual)
        self.manual_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=8)
        self.manual_view = Gtk.TextView(monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=10,
                                        bottom_margin=10, left_margin=10, right_margin=10)
        self.manual_view.get_buffer().set_text(c.get("manual_args") or core.manual_default_text())
        msw = Gtk.ScrolledWindow(min_content_height=260, css_classes=["card"])
        msw.set_child(self.manual_view)
        self.manual_box.append(msw)
        self.manual_hint = Gtk.Label(xalign=0, wrap=True, css_classes=["dim-label", "caption"])
        self.manual_box.append(self.manual_hint)
        mb = Gtk.Box(spacing=8, halign=Gtk.Align.END)
        self.clean_btn = Gtk.Button(label="Clean up", css_classes=["pill"], visible=False,
                                    tooltip_text="Replace hidden spaces, zero-width characters and curly "
                                                 "quotes/dashes with plain ones")
        self.clean_btn.connect("clicked", lambda *_: self.manual_view.get_buffer().set_text(
            core.manual_clean(self.manual_text())))
        mb.append(self.clean_btn)
        reset = Gtk.Button(label="Reset to Inkwell's command", css_classes=["pill"])
        reset.connect("clicked", lambda *_: self.manual_view.get_buffer().set_text(core.manual_default_text()))
        mapply = Gtk.Button(label="Apply & restart model", css_classes=["suggested-action", "pill"])
        mapply.connect("clicked", self.apply_manual)
        mb.append(reset)
        mb.append(mapply)
        self.manual_box.append(mb)
        mg.add(self.manual_box)
        self.manual.connect("notify::active", self.on_manual_toggle)
        self.manual_view.get_buffer().connect("changed", lambda *_: self.sync_manual_ui()
                                              if hasattr(self, "extra") else None)

        s = Adw.PreferencesGroup(title="Settings",
                                 description="Changes apply when you press Apply (the model restarts).")
        self.add(s)
        self.settings_group = s
        self.ctx = combo_row("Context size", "How much text it can keep in mind", CTX_OPTS, c["ctx"])
        self.ub = combo_row("Batch size", "Lower uses less VRAM but reads long prompts slower", UB_OPTS, c["ubatch"])
        self.kv = combo_row("Memory cache", None, KV_OPTS, c["kv_quant"])
        self.vision = combo_row("Image input", "Where the vision part runs", VISION_OPTS, c["vision_mode"])
        if not Path(c["mmproj"]).exists():
            self.vision.set_sensitive(False)
            self.vision.set_subtitle("Vision file not found: " + c["mmproj"])
        self.threads = Adw.SpinRow.new_with_range(1, 24, 1)
        self.threads.set_title("CPU threads")
        self.threads.set_value(c["threads"])
        self.auto = Adw.SwitchRow(title="Start the model when the app opens", active=c["autostart"])
        self.draft = Adw.SwitchRow(title="Speed-up drafter (DSpark)",
                                   subtitle="A tiny helper model guesses ahead: up to ~1.7× faster on code, can be "
                                            "slower on prose. Needs ~0.75 GB extra VRAM (16 GB+ cards)",
                                   active=bool(c.get("draft_on")))
        self.draft_n = Adw.SpinRow.new_with_range(2, 5, 1)
        self.draft_n.set_title("Draft length")
        self.draft_n.set_subtitle("Tokens guessed per step. 4 measured best on code; 2–3 for mostly prose")
        self.draft_n.set_value(int(c.get("draft_n", 4)))
        self.draft.bind_property("active", self.draft_n, "sensitive", 2)  # SYNC_CREATE
        self.draft.connect("notify::active", self.on_draft_toggle)
        for w in (self.ctx, self.ub, self.kv, self.vision, self.draft, self.draft_n, self.threads, self.auto):
            s.add(w)
        apply_row = Adw.ActionRow()
        ab = Gtk.Button(label="Apply & restart model", css_classes=["suggested-action", "pill"],
                        valign=Gtk.Align.CENTER, halign=Gtk.Align.END)
        ab.connect("clicked", self.apply)
        apply_row.add_suffix(ab)

        th = self.think_group = Adw.PreferencesGroup(
            title="Thinking",
            description="Defaults for everything that uses Inkling: this chat, dsh and your phone. "
                        "Apps that ask for their own effort level still get it.")
        self.add(th)
        erow = Adw.ActionRow(title="Default effort", subtitle="How hard it thinks before answering")
        self.effort_val = Gtk.Label(css_classes=["numeric", "dim-label"], width_chars=12, xalign=1)
        self.effort = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 0.99, 0.01)
        self.effort.set_size_request(300, -1)
        self.effort.set_draw_value(False)
        for v, name in EFFORT_MARKS:
            self.effort.add_mark(v, Gtk.PositionType.BOTTOM, name)
        self.effort.connect("value-changed", lambda sc: self.effort_val.set_text(
            f"{effort_name(sc.get_value())} ({sc.get_value():.2f})"))
        self.effort.set_value(float(c["think_effort"]))
        ebox = Gtk.Box(spacing=8, valign=Gtk.Align.CENTER)
        ebox.append(self.effort)
        ebox.append(self.effort_val)
        erow.add_suffix(ebox)
        th.add(erow)
        self.budget = combo_row("Thinking limit",
                                "Cuts thinking off after this many tokens and makes it answer",
                                BUDGET_OPTS, int(c["think_budget"]))
        th.add(self.budget)
        self.preserve = Adw.SwitchRow(title="Remember earlier thinking",
                                      subtitle="Keeps past thoughts in the conversation. Off saves context "
                                               "in long chats and dsh sessions",
                                      active=c["think_preserve"])
        th.add(self.preserve)
        th.add(Adw.ActionRow(title="Note", subtitle="Even on Off, Inkling still thinks a line or two. "
                                                    "The model can't fully skip it.",
                             css_classes=["dim-label"]))

        ag = Adw.PreferencesGroup()
        self.apply_group = ag
        self.add(ag)
        ag.add(apply_row)

        su = Adw.PreferencesGroup(title="Setup")
        self.add(su)
        self.engine_row = Adw.ComboRow(title="Engine", subtitle="Switch between your llama.cpp builds; "
                                                                "the model restarts on the new one")
        self.engine_keys = []
        self._engine_lock = False
        self.engine_row.connect("notify::selected", self.on_engine)
        su.add(self.engine_row)
        self.setup_row = Adw.ActionRow(title="Build path", subtitle_lines=2, subtitle_selectable=True)
        rb = Gtk.Button(label="Build another…", valign=Gtk.Align.CENTER, css_classes=["flat"],
                        tooltip_text="Build llama.cpp with another backend (CUDA, Vulkan, ROCm, CPU) or a newer "
                                     "version. Existing builds are kept so you can switch back")
        rb.connect("clicked", lambda *_: self.win.get_application().open_wizard(rebuild_only=True))
        self.setup_row.add_suffix(rb)
        su.add(self.setup_row)
        mrow = Adw.ActionRow(title="Model file", subtitle=c["model"], subtitle_lines=2)
        self.model_file_row = mrow
        mb = Gtk.Button(label="Change…", valign=Gtk.Align.CENTER, css_classes=["flat"])
        mb.connect("clicked", lambda *_: self.win.get_application().open_wizard(model_only=True))
        mrow.add_suffix(mb)
        su.add(mrow)

        a = Adw.PreferencesGroup(title="Advanced")
        self.add(a)
        self.ngl = Adw.SpinRow.new_with_range(0, 99, 1)
        self.ngl.set_title("GPU layers")
        self.ngl.set_subtitle("99 = everything on the GPU")
        self.ngl.set_value(c["gpu_layers"])
        a.add(self.ngl)
        self.cpu_layers = Adw.SpinRow.new_with_range(0, 12, 1)
        self.cpu_layers.set_title("Layers kept in system RAM")
        self.cpu_layers.set_subtitle("Moves some experts off the graphics card. Frees VRAM, costs speed. "
                                     "Set automatically by setup")
        self.cpu_layers.set_value(c.get("cpu_layers", 0))
        a.add(self.cpu_layers)
        self.extra = Adw.EntryRow(title="Extra llama-server flags")
        self.extra.set_text(c["extra_args"])
        a.add(self.extra)
        logexp = Adw.ExpanderRow(title="Server log")
        self.log = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.CHAR, css_classes=["log"])
        lsw = Gtk.ScrolledWindow(min_content_height=260)
        lsw.set_child(self.log)
        self.log_sw = lsw
        logexp.add_row(lsw)
        a.add(logexp)

    def do_map(self):
        Adw.PreferencesPage.do_map(self)
        self.sync_manual_ui()

    def refresh_engines(self):
        eng = core.engines()
        keys = list(eng)
        cur = core.current_engine()
        if keys != self.engine_keys:
            self._engine_lock = True
            self.engine_keys = keys
            self.engine_row.set_model(Gtk.StringList.new([core.engine_label(k) for k in keys] or ["None built"]))
            self._engine_lock = False
        if cur in keys and self.engine_row.get_selected() != keys.index(cur):
            self._engine_lock = True
            self.engine_row.set_selected(keys.index(cur))
            self._engine_lock = False
        self.engine_row.set_sensitive(len(keys) > 1)

    def on_engine(self, row, _p):
        if self._engine_lock or not self.engine_keys:
            return
        b = self.engine_keys[row.get_selected()]
        if b == core.current_engine():
            return

        def work():
            msg = core.switch_engine(b)
            GLib.idle_add(self.win.toast, f"Switched to {core.engine_label(b)}. {msg}…")
        threading.Thread(target=work, daemon=True).start()

    def on_draft_toggle(self, row, _p):
        """Turning the drafter on without the file downloads it next to the model."""
        c = core.CONF
        if not row.get_active() or (c.get("draft_model") and Path(c["draft_model"]).exists()):
            return
        from . import builder, catalog
        dest = Path(c.get("models_dir") or Path(c["model"]).parent) / catalog.DSPARK[0]
        row.set_subtitle("Downloading the drafter (402 MB)…")
        r = builder.Runner(lambda t: None)

        def prog(done, total, bps):
            GLib.idle_add(row.set_subtitle, f"Downloading the drafter… {done / total * 100:.0f}%")

        def work():
            try:
                path = builder.download(r, catalog.dspark_url(), dest, catalog.DSPARK[1], prog)
                c["draft_model"] = path
                core.save_conf()
                GLib.idle_add(row.set_subtitle, "Downloaded. Press Apply to use it")
            except Exception as e:
                GLib.idle_add(row.set_subtitle, f"Download failed: {e}")
                GLib.idle_add(row.set_active, False)
        threading.Thread(target=work, daemon=True).start()

    def manual_text(self):
        b = self.manual_view.get_buffer()
        return b.get_text(b.get_start_iter(), b.get_end_iter(), False)

    def sync_manual_ui(self):
        on = self.manual.get_active()
        self.manual_box.set_visible(on)
        for w in (self.settings_group, self.think_group, self.apply_group, self.cpu_layers, self.ngl, self.extra):
            w.set_sensitive(not on)
        self.settings_group.set_description("Ignored in Manual mode." if on else
                                            "Changes apply when you press Apply (the model restarts).")
        try:
            argv = core.manual_argv(self.manual_text())
            h, p = "127.0.0.1", 8080
            for i, a in enumerate(argv):
                k, _, v = a.partition("=")
                v = v or (argv[i + 1] if i + 1 < len(argv) else "")
                if k == "--port" and v.isdigit():
                    p = int(v)
                if k == "--host" and v not in ("0.0.0.0", "::", "[::]"):
                    h = v
            notes = [f"Inkwell will talk to it at http://{h}:{p}" + ("" if any(a.split("=")[0] == "--port" for a in argv)
                                                                     else " (llama-server's default; add --port to change)")]
            if not any(a in ("-m", "--model", "-hf", "--hf-repo") or a.startswith(("--model=", "-hf=")) for a in argv):
                notes.append("⚠ No -m/--model given")
            issues = core.manual_issues(self.manual_text())
            self.clean_btn.set_visible(any("empty argument" not in m for _, m in issues))
            if issues:
                self.manual_hint.add_css_class("warning")
                self.manual_hint.set_text("⚠ " + "; ".join(f"line {n}: {m}" for n, m in issues[:6])
                                          + (" …" if len(issues) > 6 else "") + "\n" + " · ".join(notes))
                return
            self.manual_hint.remove_css_class("warning")
            notes.append("One flag per line is easiest; # starts a comment.")
            self.manual_hint.set_text(" · ".join(notes))
        except ValueError as e:
            self.manual_hint.set_text(f"⚠ Can't read this: {e}")

    def on_manual_toggle(self, *_):
        self.sync_manual_ui()
        if not self.manual.get_active() and core.CONF.get("manual"):
            core.CONF["manual"] = False
            core.save_conf()
            threading.Thread(target=lambda: (core.stop(), core.start(),
                                             GLib.idle_add(self.win.toast, "Back to Inkwell's settings. Restarting…")),
                             daemon=True).start()

    def apply_manual(self, *_):
        text = self.manual_text()
        fixed = [m for _, m in core.manual_issues(text) if "empty argument" not in m]
        if fixed:
            text = core.manual_clean(text)
            self.manual_view.get_buffer().set_text(text)
        empty = [n for n, m in core.manual_issues(text) if "empty argument" in m]
        if empty:
            return self.win.toast(f"Line {empty[0]} has an empty argument (\"\" or ''). Remove it first")
        try:
            core.manual_argv(text)
        except ValueError as e:
            return self.win.toast(f"Can't read the flags: {e}")
        if fixed:
            self.win.toast(f"Cleaned up {len(fixed)} hidden/look-alike character issue(s)")
        core.CONF["manual"], core.CONF["manual_args"] = True, text
        core.save_conf()

        def work():
            core.stop()
            msg = core.start()
            GLib.idle_add(self.win.toast, f"Manual mode: {msg}…")
        threading.Thread(target=work, daemon=True).start()

    def apply(self, *_):
        c = core.CONF
        c["ctx"] = self.ctx.keys[self.ctx.get_selected()]
        c["ubatch"] = self.ub.keys[self.ub.get_selected()]
        c["kv_quant"] = self.kv.keys[self.kv.get_selected()]
        c["vision_mode"] = self.vision.keys[self.vision.get_selected()]
        c["threads"] = int(self.threads.get_value())
        c["autostart"] = self.auto.get_active()
        c["gpu_layers"] = int(self.ngl.get_value())
        c["extra_args"] = self.extra.get_text()
        c["cpu_layers"] = int(self.cpu_layers.get_value())
        c["draft_on"] = self.draft.get_active()
        c["draft_n"] = int(self.draft_n.get_value())
        c["think_effort"] = round(self.effort.get_value(), 2)
        c["think_budget"] = self.budget.keys[self.budget.get_selected()]
        c["think_preserve"] = self.preserve.get_active()
        core.save_conf()

        def work():
            core.stop()
            msg = core.start()
            GLib.idle_add(self.win.toast, f"Saved. {msg}…")
        threading.Thread(target=work, daemon=True).start()

    def update(self, st):
        run, h = st["running"], st["health"]
        if not run:
            self.status_row.set_subtitle("Crashed. Check the server log below" if st["exit"] else "Stopped")
        else:
            self.status_row.set_subtitle("Ready" if h == "ok" else "Loading…")
        self.start_btn.set_visible(not run)
        self.stop_btn.set_visible(run)
        c = core.CONF
        self.head.set_description("Manual mode · running your own flags" if c.get("manual") else
                                  Path(c["model"]).name if c["model"] else "No model set up")
        self.model_file_row.set_subtitle(c["model"] or "Not set")
        self.setup_row.set_subtitle(f"{c.get('llama_server') or 'not built yet'}")
        self.refresh_engines()
        g = st["gpu"]
        if g:
            self.vram_lbl.set_text(f"{g['used'] / 1024:.1f} / {g['total'] / 1024:.0f} GB")
            self.vram_bar.set_value(g["used"] / g["total"])
            self.vram_row.set_subtitle(g["name"] + (f" · Inkling uses {g['model'] / 1024:.1f} GB" if g["model"] else ""))
        has = core.engine_has_dspark()
        self.draft.set_sensitive(has and not core.CONF.get("manual"))
        if not has:
            self.draft.set_subtitle("This engine was built without drafter support. Use Setup → Build another… "
                                    "to build one with it")
        elif self.draft.get_subtitle().startswith("This engine"):
            self.draft.set_subtitle("A tiny helper model guesses ahead: up to ~1.7× faster on code, can be slower "
                                    "on prose. Needs ~0.75 GB extra VRAM (16 GB+ cards)")
        # Follow the saved setting only when it changes behind our back (e.g. the drafter watchdog turned it
        # off). Never overwrite a switch the user just flipped and hasn't applied yet.
        saved = bool(core.CONF.get("draft_on"))
        if saved != getattr(self, "_draft_saved", saved) and self.draft.get_active() != saved:
            self.draft.set_active(saved)
        self._draft_saved = saved
        lv = core.activity.live()
        if lv["state"] == "writing":
            self.speed_row.set_subtitle(f"Writing now at {lv['gen_tps']:.1f} tok/s for {lv['source']}")
        elif lv["state"] == "reading":
            self.speed_row.set_subtitle(f"Reading a {lv['prompt_n']:,}-token prompt for {lv['source']}")
        elif lv["last"]:
            l = lv["last"]
            self.speed_row.set_subtitle(f"Last: {l['tps']:.1f} tok/s writing"
                                        + (f", {l['prompt_tps']:.0f} tok/s reading" if l["prompt"] > 1 else "")
                                        + f" · {l['source']} · {ago(l['time'])}")
        buf = self.log.get_buffer()
        if buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False) != st["log"]:
            adj = self.log_sw.get_vadjustment()
            at_end = adj.get_value() + adj.get_page_size() >= adj.get_upper() - 10
            buf.set_text(st["log"])
            if at_end:
                GLib.idle_add(lambda: adj.set_value(adj.get_upper()) or False)


# ---------------- network page ----------------
def copy_row(title, win):
    row = Adw.ActionRow(title=title, subtitle_selectable=True)
    row.add_css_class("property")
    b = Gtk.Button(icon_name="edit-copy-symbolic", tooltip_text="Copy", css_classes=["flat"], valign=Gtk.Align.CENTER)
    b.connect("clicked", lambda *_: (copy_text(row, row.get_subtitle()), win.toast("Copied")))
    row.add_suffix(b)
    return row


class NetworkPage(Adw.PreferencesPage):
    def __init__(self, win):
        super().__init__()
        self.win = win
        g = Adw.PreferencesGroup(
            title="Phone &amp; other devices",
            description="Shares Inkling on your home Wi-Fi as an OpenAI-compatible API. In your phone's chat app, "
                        "add a custom OpenAI provider and paste in these three values.")
        self.add(g)
        self.lan = Adw.SwitchRow(title="Share on Wi-Fi", active=core.CONF["lan"])
        self.lan.connect("notify::active", lambda r, _p: core.set_lan(r.get_active()))
        g.add(self.lan)
        self.url_row = copy_row("Base URL", win)
        self.key_row = copy_row("API key", win)
        regen = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Make a new key (old one stops working)",
                           css_classes=["flat"], valign=Gtk.Align.CENTER)
        regen.connect("clicked", lambda *_: (core.new_token(), win.toast("New key made. Update it on your phone.")))
        self.key_row.add_suffix(regen)
        self.model_row = copy_row("Model", win)
        for r in (self.url_row, self.key_row, self.model_row):
            g.add(r)
        qbox = Gtk.Box(spacing=24, halign=Gtk.Align.CENTER, margin_top=12)
        self.qr_url, self.qr_key = QRCodeArea(), QRCodeArea()
        for q, cap in ((self.qr_url, "Scan: Base URL"), (self.qr_key, "Scan: API key")):
            b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            b.append(q)
            b.append(Gtk.Label(label=cap, css_classes=["dim-label", "caption"]))
            qbox.append(b)
        self.qbox = qbox
        g.add(qbox)
        self._qr_for = None

        d = Adw.PreferencesGroup(title="dsh (DeepSeek Harness)",
                                 description="Adds Inkling to a dsh profile. A backup of the profile is saved "
                                             "first. Restart dsh afterwards.")
        self.add(d)
        self.dsh_group = d
        profiles = core.dsh_profiles() or ["web"]
        self.dsh_prof = Adw.ComboRow(title="Profile")
        self.dsh_prof.set_model(Gtk.StringList.new(profiles))
        cur = core.CONF.get("dsh_profile", "web")
        self.dsh_prof.set_selected(profiles.index(cur) if cur in profiles else 0)

        def prof_changed(r, _p):
            core.CONF["dsh_profile"] = profiles[r.get_selected()]
            core.save_conf()
        self.dsh_prof.connect("notify::selected", prof_changed)
        d.add(self.dsh_prof)
        self.dsh_row = Adw.ActionRow(title="Status")
        d.add(self.dsh_row)
        brow = Adw.ActionRow()
        b1 = Gtk.Button(label="Connect & make default", css_classes=["suggested-action", "pill"], valign=Gtk.Align.CENTER)
        b2 = Gtk.Button(label="Only add as an option", css_classes=["pill"], valign=Gtk.Align.CENTER)
        b1.connect("clicked", lambda *_: self.dsh(True))
        b2.connect("clicked", lambda *_: self.dsh(False))
        brow.add_suffix(b2)
        brow.add_suffix(b1)
        d.add(brow)

        l = Adw.PreferencesGroup(title="Apps on this PC",
                                 description="OpenCode, Aider, Continue and others: pick an \"OpenAI-compatible\" "
                                             "provider. No API key needed.")
        self.add(l)
        self.local_row = copy_row("Base URL", win)
        self.local_row.set_subtitle(core.llama_url("/v1"))
        l.add(self.local_row)
        m2 = copy_row("Model", win)
        m2.set_subtitle(core.MODEL_ID)
        l.add(m2)

    def dsh(self, default):
        try:
            backup = core.dsh_connect(default)
            self.win.toast(f"dsh updated (backup: {backup})")
        except Exception as e:
            self.win.toast(f"Couldn't update dsh: {e}")

    def update(self, st):
        self.local_row.set_subtitle(core.llama_url("/v1"))
        on = core.CONF["lan"]
        url = f"http://{st['ip']}:{core.LAN_PORT}/v1"
        self.url_row.set_subtitle(url)
        self.key_row.set_subtitle(core.CONF["token"])
        self.model_row.set_subtitle(core.MODEL_ID)
        for r in (self.url_row, self.key_row, self.model_row, self.qbox):
            r.set_visible(on)
        if core.lan.error:
            self.lan.set_subtitle("Couldn't open port: " + core.lan.error)
        else:
            self.lan.set_subtitle(f"Only devices with the key can use it · port {core.LAN_PORT}" if on else "Off")
        if on and self._qr_for != (url, core.CONF["token"]):
            self.qr_url.set_text(url)
            self.qr_key.set_text(core.CONF["token"])
            self._qr_for = (url, core.CONF["token"])
        d = st["dsh"]
        if not d.get("installed"):
            txt = "dsh isn't installed (or has no profiles yet)"
        elif d.get("error"):
            txt = "Couldn't read the dsh config: " + d["error"]
        elif d.get("is_default"):
            txt = "Connected · Inkling is dsh's default model"
        elif d.get("provider"):
            df = d.get("default") or {}
            txt = f"Connected · dsh's default is still {df.get('provider')}/{df.get('model')}"
        else:
            df = d.get("default") or {}
            txt = f"Not connected · dsh currently uses {df.get('provider')}/{df.get('model')}"
        self.dsh_row.set_subtitle(txt)


# ---------------- activity dropdown ----------------
KIND_ICON = {"start": "mail-send-symbolic", "done": "emblem-ok-symbolic", "warn": "dialog-warning-symbolic",
             "error": "dialog-error-symbolic", "stop": "media-playback-stop-symbolic",
             "info": "content-loading-symbolic", "ok": "emblem-ok-symbolic"}


def live_text(lv):
    if lv["state"] == "reading":
        return (f"Reading prompt · {lv['prompt_frac'] * 100:.0f}% of {lv['prompt_n']:,} tokens"
                + (f" · {lv['prompt_tps']:.0f} tok/s" if lv["prompt_tps"] else ""))
    if lv["state"] == "writing":
        return f"Writing · {lv['gen_tps']:.1f} tok/s · {lv['gen_n']:,} tokens so far"
    return "Idle"


class ActivityView(Gtk.Box):
    """Live status + plain-English feed (or the raw server log), newest at the bottom like a terminal."""

    def __init__(self, win, compact):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.win = win
        self.shown = []            # events currently in the list
        status = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, css_classes=["card"], margin_bottom=2)
        self.status = Gtk.Label(xalign=0, css_classes=["title-4", "numeric"], margin_top=8, margin_start=12, margin_end=12)
        self.source = Gtk.Label(xalign=0, css_classes=["dim-label", "caption"], margin_bottom=8, margin_start=12,
                                margin_end=12, ellipsize=Pango.EllipsizeMode.END)
        status.append(self.status)
        status.append(self.source)
        self.append(status)

        bar = Gtk.Box(spacing=6)
        self.mode_feed = Gtk.ToggleButton(label="What it's doing", active=True)
        self.mode_raw = Gtk.ToggleButton(label="Raw log", group=self.mode_feed)
        seg = Gtk.Box(css_classes=["linked"])
        seg.append(self.mode_feed)
        seg.append(self.mode_raw)
        bar.append(seg)
        bar.append(Gtk.Box(hexpand=True))
        cp = Gtk.Button(icon_name="edit-copy-symbolic", css_classes=["flat"],
                        tooltip_text="Copy the full technical log (to send to someone)")
        cp.connect("clicked", lambda *_: (copy_text(self, core.log_tail(3000)), self.win.toast("Log copied")))
        bar.append(cp)
        self.extra = bar
        self.append(bar)

        self.stack = Gtk.Stack(vexpand=True, transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.list = Gtk.ListBox(css_classes=["boxed-list"], selection_mode=Gtk.SelectionMode.NONE, valign=Gtk.Align.END)
        self.feed_sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        self.feed_sw.set_child(self.list)
        self.raw = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.CHAR, css_classes=["log"],
                                top_margin=6, bottom_margin=6, left_margin=6, right_margin=6)
        self.raw_sw = Gtk.ScrolledWindow(vexpand=True)
        self.raw_sw.set_child(self.raw)
        if compact:
            for sw in (self.feed_sw, self.raw_sw):
                sw.set_min_content_height(340)
                sw.set_max_content_height(340)
        self.stack.add_named(self.feed_sw, "feed")
        self.stack.add_named(self.raw_sw, "raw")
        self.append(self.stack)
        self.mode_feed.connect("toggled", lambda b: b.get_active() and self.stack.set_visible_child_name("feed"))
        self.mode_raw.connect("toggled", lambda b: b.get_active() and (self.stack.set_visible_child_name("raw"),
                                                                       self.refresh(force=True)))
        self.refresh(force=True)

    @staticmethod
    def _at_end(sw):
        a = sw.get_vadjustment()
        return a.get_value() + a.get_page_size() >= a.get_upper() - 30

    @staticmethod
    def _to_end(sw):
        GLib.idle_add(lambda: sw.get_vadjustment().set_value(sw.get_vadjustment().get_upper()) or False)

    def _row(self, t, kind, text):
        row = Adw.ActionRow(title=GLib.markup_escape_text(text), title_lines=5,
                            subtitle=time.strftime("%H:%M:%S", time.localtime(t)))
        row.add_prefix(Gtk.Image(icon_name=KIND_ICON.get(kind, "dialog-information-symbolic")))
        if kind in ("warn", "error"):
            row.add_css_class("warning" if kind == "warn" else "error")
        return row

    def refresh(self, force=False):
        lv = core.activity.live()
        if not core.running():
            self.status.set_text("Model stopped")
        else:
            self.status.set_text(live_text(lv))
        self.source.set_text(f"For: {lv['source']}" if lv["state"] != "idle" else
                             (f"Last: {lv['last']['tps']:.1f} tok/s for {lv['last']['source']}, {ago(lv['last']['time'])}"
                              if lv["last"] else "Waiting for a request from the chat, dsh or your phone"))
        feed = core.activity.feed()
        end = self._at_end(self.feed_sw)
        if force or feed[:len(self.shown)] != self.shown:
            while (r := self.list.get_first_child()):
                self.list.remove(r)
            self.shown = []
        for e in feed[len(self.shown):]:
            self.list.append(self._row(*e))
        if len(feed) != len(self.shown) or force:
            self.shown = list(feed)
            if end or force:
                self._to_end(self.feed_sw)
        if self.stack.get_visible_child_name() == "raw":
            txt = core.log_tail(400)
            buf = self.raw.get_buffer()
            if buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False) != txt:
                end = self._at_end(self.raw_sw)
                buf.set_text(txt)
                if end or force:
                    self._to_end(self.raw_sw)


class ActivityWindow(Adw.Window):
    """Pop-out live view to keep next to dsh."""

    def __init__(self, win):
        super().__init__(title="Inkwell · Live activity", default_width=440, default_height=760)
        tv = Adw.ToolbarView()
        tv.add_top_bar(Adw.HeaderBar())
        self.view = ActivityView(win, compact=False)
        for m in ("top", "bottom", "start", "end"):
            getattr(self.view, f"set_margin_{m}")(10)
        tv.set_content(self.view)
        self.set_content(tv)


class ActivityButton(Gtk.MenuButton):
    """Header dropdown with the live view and a button to pop it out into its own window."""

    def __init__(self, win):
        super().__init__(icon_name="view-list-bullet-symbolic", tooltip_text="Activity: what Inkling is doing")
        self.win = win
        self.seen = 0
        self.popout = None
        pop = Gtk.Popover()
        self.set_popover(pop)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=8, margin_bottom=8,
                      margin_start=8, margin_end=8)
        box.set_size_request(500, -1)
        head = Gtk.Box()
        head.append(Gtk.Label(label="Activity", css_classes=["heading"], hexpand=True, xalign=0))
        pb = Gtk.Button(label="Pop out", icon_name="window-new-symbolic", css_classes=["flat"],
                        tooltip_text="Open in its own window, e.g. next to dsh")
        pb.set_child(Adw.ButtonContent(label="Pop out", icon_name="window-new-symbolic"))
        pb.connect("clicked", lambda *_: (pop.popdown(), self.open_window()))
        head.append(pb)
        box.append(head)
        self.view = ActivityView(win, compact=True)
        box.append(self.view)
        pop.set_child(box)
        pop.connect("show", lambda *_: self.view.refresh(force=True))

    def open_window(self):
        if not self.popout:
            self.popout = ActivityWindow(self.win)
            self.popout.connect("close-request", lambda *_: setattr(self, "popout", None) or False)
        self.popout.present()

    def refresh(self):
        feed_len = len(core.activity.feed())
        visible = self.get_popover().get_visible()
        if visible or self.popout:
            self.seen = feed_len
        unseen = [e for e in core.activity.feed()[self.seen:] if e[1] in ("warn", "error")]
        self.set_icon_name("dialog-warning-symbolic" if unseen else "view-list-bullet-symbolic")
        if visible:
            self.view.refresh()
        if self.popout:
            self.popout.view.refresh()


def ago(t):
    d = time.time() - t
    return "just now" if d < 60 else f"{int(d // 60)} min ago" if d < 3600 else f"{int(d // 3600)} h ago"


# ---------------- window / app ----------------
class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Inkwell", default_width=1100, default_height=760)
        self.last_speed = ""
        self.toasts = Adw.ToastOverlay()
        tv = Adw.ToolbarView()
        hb = Adw.HeaderBar()
        self.stack = Adw.ViewStack()
        sw = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        hb.set_title_widget(sw)
        self.pill = Gtk.Label(label="…", css_classes=["status-pill", "numeric"])
        self.activity = ActivityButton(self)
        hb.pack_end(self.activity)
        hb.pack_end(self.pill)
        tv.add_top_bar(hb)
        self.chat = ChatPage(self)
        self.model = ModelPage(self)
        self.net = NetworkPage(self)
        self.stack.add_titled_with_icon(self.chat, "chat", "Chat", "chat-message-new-symbolic")
        self.stack.add_titled_with_icon(self.model, "model", "Model", "application-x-executable-symbolic")
        self.stack.add_titled_with_icon(self.net, "net", "Network & dsh", "network-wireless-symbolic")
        tv.set_content(self.stack)
        self.toasts.set_child(tv)
        self.set_content(self.toasts)
        self.connect("close-request", self.on_close)
        self._polling = False
        GLib.timeout_add(1000, self.poll)
        self.poll()

    def toast(self, msg):
        self.toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(str(msg)), timeout=3))
        return False

    def poll(self):
        if self._polling:
            return True
        self._polling = True

        def work():
            core.activity.poll()
            st = {"running": core.running(), "exit": core.exit_code(), "gpu": core.gpu(),
                  "log": core.log_tail(), "ip": core.lan_ip(), "dsh": core.dsh_state()}
            st["health"] = core.health() if st["running"] else None
            GLib.idle_add(self.apply_status, st)
        threading.Thread(target=work, daemon=True).start()
        return True

    def apply_status(self, st):
        self._polling = False
        p = self.pill
        for c in ("status-ok", "status-warn", "status-err"):
            p.remove_css_class(c)
        if not st["running"]:
            p.set_text("● Crashed" if st["exit"] else "● Stopped")
            p.add_css_class("status-err" if st["exit"] else "status-warn")
        elif st["health"] == "ok":
            lv = core.activity.live()
            if lv["state"] == "reading":
                p.set_text(f"● Reading prompt {lv['prompt_frac'] * 100:.0f}%"
                           + (f" · {lv['prompt_tps']:.0f} tok/s" if lv["prompt_tps"] else ""))
            elif lv["state"] == "writing":
                p.set_text(f"● Writing · {lv['gen_tps']:.1f} tok/s")
            else:
                p.set_text("● Ready" + (" · Wi-Fi on" if core.CONF["lan"] else ""))
            p.set_tooltip_text(f"Working for: {lv['source']}" if lv["state"] != "idle" else None)
            p.add_css_class("status-ok")
        else:
            p.set_text("● Loading…")
            p.add_css_class("status-warn")
        self.model.update(st)
        self.net.update(st)
        self.activity.refresh()
        return False

    def on_close(self, *_):
        app = self.get_application()
        if not core.running():
            app.quit_all()
            return False
        dlg = Adw.AlertDialog(heading="Keep Inkling running?",
                              body="dsh and your phone can only use Inkling while it's running. Open "
                                   "Inkwell again to bring this window back.")
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("quit", "Stop & quit")
        dlg.add_response("bg", "Keep running in background")
        dlg.set_response_appearance("bg", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("bg")
        dlg.set_close_response("cancel")

        def resp(d, r):
            if r == "quit":
                app.quit_all()
            elif r == "bg":
                if not app.held:
                    app.hold()
                    app.held = True
                self.set_visible(False)
        dlg.connect("response", resp)
        dlg.present(self)
        return True


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.win = None
        self.held = False

    def do_startup(self):
        Adw.Application.do_startup(self)
        css = Gtk.CssProvider()
        css.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        src_icons = Path(__file__).resolve().parent.parent / "data"  # running from a source checkout
        if src_icons.exists():
            Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).add_search_path(str(src_icons))
        Gtk.Window.set_default_icon_name(APP_ID)
        core.lan.set(core.CONF["lan"])
        if core.CONF["autostart"] and core.CONF.get("setup_done"):
            core.start()

    def do_activate(self):
        if not core.CONF.get("setup_done"):
            if not getattr(self, "wiz", None):
                self.open_wizard()
            else:
                self.wiz.present()
            return
        if not self.win:
            self.win = Window(self)
        self.win.set_visible(True)
        self.win.present()

    def open_wizard(self, rebuild_only=False, model_only=False):
        def done():
            self.wiz = None
            if rebuild_only or model_only:
                threading.Thread(target=lambda: (core.stop(), core.start()), daemon=True).start()
            elif core.CONF["autostart"]:
                core.start()  # first run: startup skipped autostart because setup wasn't done yet
            self.activate()
        if rebuild_only or model_only:
            threading.Thread(target=core.stop, daemon=True).start()
        self.wiz = wizard.Wizard(self, done, rebuild_only=rebuild_only, model_only=model_only)
        self.wiz.set_modal(bool(self.win))
        if self.win:
            self.wiz.set_transient_for(self.win)
        self.wiz.present()

    def quit_all(self):
        core.stop()
        core.lan.set(False)
        self.quit()

    def do_shutdown(self):
        core.stop()
        Adw.Application.do_shutdown(self)


def main():
    sys.exit(App().run(sys.argv))


if __name__ == "__main__":
    main()
