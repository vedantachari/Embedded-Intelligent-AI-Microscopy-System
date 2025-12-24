import threading
from kivy.clock import mainthread
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle
from kivy.properties import ListProperty
from kivy.uix.behaviors import ButtonBehavior


class RoundedButton(ButtonBehavior, BoxLayout):
    bg_color = ListProperty([1, 1, 1, 1])
    text_color = ListProperty([0, 0, 0, 1])

    def __init__(self, text="", bg_color=(1,1,1,1), text_color=(0,0,0,1), radius=12, **kwargs):
        super().__init__(orientation="horizontal", **kwargs)
        self.padding = (dp(10), dp(8))
        self.spacing = dp(6)
        self.size_hint_y = None
        self.height = dp(44)
        self.radius = radius
        self.bg_color = bg_color
        self.text_color = text_color

        from kivy.graphics import RoundedRectangle
        with self.canvas.before:
            self._bg_color_inst = Color(*self.bg_color)
            self._bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])
        self._label = Label(text=text, color=self.text_color, halign="center", valign="middle", markup=False)
        self._label.text_size = (None, None)
        self.add_widget(self._label)

        self.bind(pos=self._update_rect, size=self._update_rect)
        self.bind(bg_color=self._update_color, text_color=self._update_text_color)

    def _update_rect(self, *a):
        try:
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size
        except Exception:
            pass

    def _update_color(self, instance, value):
        try:
            self._bg_color_inst.rgba = value
        except Exception:
            pass

    def _update_text_color(self, instance, value):
        try:
            self._label.color = value
        except Exception:
            pass

    @property
    def text(self):
        return self._label.text

    @text.setter
    def text(self, v):
        self._label.text = v


class InfoPopupHelper:
    def __init__(self, parent_widget, gemini_client):
        self.parent = parent_widget
        self.client = gemini_client
        self._popup = None

    def show_for_class(self, cls_idx: int, cls_name: str, extra_context: str = None):
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        loading_label = Label(
            text=f"Fetching detailed info for '{cls_name}'...",
            size_hint_y=None, height=dp(44),
            halign="center", valign="middle", color=(0,0,0,1),
            font_size="18sp"
        )
        content.add_widget(loading_label)
        btn_close = RoundedButton(text="Cancel", bg_color=(0.9,0.9,0.9,1), text_color=(0,0,0,1))
        content.add_widget(btn_close)

        self._popup = Popup(title=f"Info: {cls_name}", content=content,
                            size_hint=(0.8, 0.8), auto_dismiss=False, background='')
        btn_close.bind(on_release=self._close_popup)
        self._popup.open()

        thread = threading.Thread(target=self._background_fetch, args=(cls_idx, cls_name, extra_context), daemon=True)
        thread.start()

    def _background_fetch(self, cls_idx, cls_name, extra_context):
        try:
            result = self.client.generate_info(class_name=cls_name, extra_context=extra_context)
        except Exception as e:
            result = {"raw": f"[Error calling Gemini API: {e}]"}
        self._show_result_on_main_thread(result, cls_name)

    @mainthread
    def _show_result_on_main_thread(self, result, fallback_name):
        if not self._popup:
            return

        card = BoxLayout(orientation="vertical", spacing=8, padding=8)

        with card.canvas.before:
            self._shadow_color = Color(0, 0, 0, 0.12)
            self._shadow_rect = Rectangle(pos=(0, 0), size=(0, 0))
            self._card_color = Color(1, 1, 1, 1)
            self._card_rect = Rectangle(pos=(0, 0), size=(0, 0))

        def _update_rects(instance, value):
            x, y = card.pos
            w, h = card.size
            self._shadow_rect.pos = (x + dp(4), y - dp(4))
            self._shadow_rect.size = (w, h)
            self._card_rect.pos = (x, y)
            self._card_rect.size = (w, h)
        card.bind(pos=_update_rects, size=_update_rects)

        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation="vertical", spacing=10, size_hint_y=None, padding=(16,16))
        content.bind(minimum_height=content.setter('height'))

        width_px = max(420, int(Window.width * 0.65))

        def make_label(text, font_size="16sp", bold=False):
            txt = f"[b]{text}[/b]" if bold else text
            lbl = Label(
                text=txt,
                size_hint_y=None,
                halign="left",
                valign="top",
                text_size=(width_px - dp(66), None),
                markup=True,
                font_size=font_size,
                color=(0,0,0,1),
                shorten=False
            )
            lbl.texture_update()
            lbl.height = lbl.texture_size[1] + dp(12)
            return lbl

        if isinstance(result, dict) and result.get("raw") and len(result.keys()) == 1:
            raw_text = result.get("raw", "")
            content.add_widget(make_label(raw_text, font_size="16sp"))
        else:
            title_text = result.get("title") or fallback_name or ""
            if title_text:
                content.add_widget(make_label(title_text, font_size="22sp", bold=True))

            summary = result.get("summary", "")
            if summary:
                content.add_widget(make_label(summary, font_size="18sp"))

            def add_section(title, items, item_font="15sp"):
                if not items:
                    return
                content.add_widget(make_label(title, font_size="17sp", bold=True))
                for it in items:
                    bullet = f"• {it}"
                    content.add_widget(make_label(bullet, font_size=item_font))

            add_section("Identification", result.get("identification", []))
            size_text = result.get("size", "")
            if size_text:
                content.add_widget(make_label(f"Size / Scale: {size_text}", font_size="15sp"))
            add_section("Microscopy Tips", result.get("microscopy_tips", []))
            add_section("Imaging Signatures", result.get("imaging_signatures", []))

            refs = result.get("references", [])
            if refs:
                content.add_widget(make_label("References", font_size="17sp", bold=True))
                for r in refs:
                    content.add_widget(make_label(r, font_size="14sp"))

            if len(content.children) == 0:
                raw = result.get("raw", "")
                content.add_widget(make_label(raw, font_size="16sp"))

        scroll.add_widget(content)
        card.add_widget(scroll)

        btns = BoxLayout(size_hint_y=None, height=dp(52), spacing=10, padding=(8,8))
        btn_copy = RoundedButton(text="Copy Text", bg_color=(0.95,0.95,0.95,1), text_color=(0,0,0,1), radius=10)
        btn_close = RoundedButton(text="Close", bg_color=(0.85,0.2,0.2,1), text_color=(1,1,1,1), radius=10)
        btns.add_widget(btn_copy)
        btns.add_widget(btn_close)
        card.add_widget(btns)

        def _close(*a):
            self._close_popup()

        def _copy(*a):
            try:
                from kivy.core.clipboard import Clipboard
                texts = []
                for child in reversed(content.children):
                    if hasattr(child, "text"):
                        txt = child.text.replace("[b]", "").replace("[/b]", "")
                        texts.append(txt)
                Clipboard.copy("\n\n".join(texts))
            except Exception:
                pass

        btn_close.bind(on_release=_close)
        btn_copy.bind(on_release=_copy)

        self._popup.content = card

    @mainthread
    def _close_popup(self, *a):
        if self._popup:
            try:
                self._popup.dismiss()
            except Exception:
                pass
            self._popup = None
