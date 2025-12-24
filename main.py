

from gemini_client import GeminiClient
from info_popup import InfoPopupHelper

import threading, time, csv, os
from collections import Counter

import cv2
from ultralytics import YOLO

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import (
    StringProperty, BooleanProperty, DictProperty, NumericProperty, ListProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.popup import Popup
from kivy.uix.label import Label
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

KV = r'''
#:import dp kivy.metrics.dp

<ModernButton>:
    size_hint_y: None
    height: dp(46)
    padding: dp(12), dp(8)
    spacing: dp(8)
    canvas.before:
        Color:
            rgba: root.shadow_color
        RoundedRectangle:
            pos: (self.x, self.y - dp(2))
            size: (self.width, self.height + dp(4))
            radius: [12,]
        Color:
            rgba: root.pressed_color if self.state == 'down' else root.bg_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [12,]
    Label:
        text: root.text
        color: 1,1,1,1
        font_size: '14sp'
        halign: 'center'
        valign: 'middle'
        text_size: self.size

<InfoButton@ButtonBehavior+BoxLayout>:
    size_hint_x: 0.12
    height: dp(36)
    padding: 0, 0
    canvas.before:
        Color:
            rgba: (.22,.22,.22,1) if self.state == 'normal' else (.17,.17,.17,1)
        RoundedRectangle:
            pos: (self.x + dp(4), self.y + dp(4))
            size: (self.width - dp(8), self.height - dp(8))
            radius: [10,]
    Label:
        text: 'i'
        font_size: '14sp'
        color: 1,1,1,1
        halign: 'center'
        valign: 'middle'
        text_size: self.size

<ClassItem>:
    name_text: ''
    count_text: ''
    cls_idx: -1
    size_hint_y: None
    height: dp(48)
    padding: dp(8), dp(6)
    spacing: dp(8)
    canvas.before:
        Color:
            rgba: 0.18,0.18,0.18,0.95
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [10,]
    Label:
        text: root.name_text
        size_hint_x: 0.64
        halign: 'left'
        valign: 'middle'
        text_size: self.size
        font_size: '16sp'
        color: 1,1,1,1
    Label:
        text: root.count_text
        size_hint_x: 0.18
        halign: 'right'
        valign: 'middle'
        text_size: self.size
        font_size: '16sp'
        color: 1,1,1,1
    InfoButton:
        id: info_btn

<VideoDashboard>:
    orientation: 'vertical'
    padding: dp(12)
    spacing: dp(12)
    canvas.before:
        Color:
            rgba: 0.1, 0.1, 0.1, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [12,]

    BoxLayout:
        size_hint_y: None
        height: dp(60)
        spacing: dp(10)

        ModernButton:
            id: btn_model
            text: root.load_model_text
            bg_color: [0.20, 0.26, 0.30, 1]
            pressed_color: [0.14, 0.18, 0.22, 1]
            shadow_color: [0,0,0,0.10]
            on_release: root.open_model_chooser()

        ModernButton:
            id: btn_video
            text: root.load_video_text
            bg_color: [0.20, 0.26, 0.30, 1]
            pressed_color: [0.14, 0.18, 0.22, 1]
            shadow_color: [0,0,0,0.10]
            on_release: root.show_video_options()

        ModernButton:
            id: btn_start
            text: root.start_stop_text
            bg_color: [0.20, 0.26, 0.30, 1]
            pressed_color: [0.14, 0.18, 0.22, 1]
            shadow_color: [0,0,0,0.10]
            on_release: root.toggle_processing()

        ModernButton:
            id: btn_save
            text: "Save CSV"
            bg_color: [0.20, 0.26, 0.30, 1]
            pressed_color: [0.14, 0.18, 0.22, 1]
            shadow_color: [0,0,0,0.10]
            on_release: root.save_csv()

    BoxLayout:
        spacing: dp(12)

        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.72
            padding: dp(8)
            canvas.before:
                Color:
                    rgba: .03, .03, .03, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10,]

            Image:
                id: video_image
                allow_stretch: True
                keep_ratio: True

        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.28
            padding: dp(8)
            spacing: dp(8)

            BoxLayout:
                size_hint_y: None
                height: dp(54)
                padding: dp(8)
                canvas.before:
                    Color:
                        rgba: .12, .12, .12, 1
                    RoundedRectangle:
                        pos: (self.x, self.y)
                        size: (self.width, self.height)
                        radius: [12,]
                Label:
                    text: root.total_text
                    color: 1,1,1,1
                    halign: 'center'
                    valign: 'middle'
                    font_size: '15sp'
                    text_size: self.size

            ScrollView:
                do_scroll_x: False
                bar_width: dp(6)
                scroll_type: ['bars', 'content']
                GridLayout:
                    id: classes_box
                    cols: 1
                    size_hint_y: None
                    height: self.minimum_height
                    row_default_height: dp(48)
                    spacing: dp(8)
'''

# ----------------- Widgets & App logic -----------------
class ModernButton(ButtonBehavior, BoxLayout):
    text = StringProperty("")
    bg_color = ListProperty([0.20, 0.26, 0.30, 1])
    pressed_color = ListProperty([0.14, 0.18, 0.22, 1])
    shadow_color = ListProperty([0, 0, 0, 0.12])

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.next_id = 0
        self.tracks = {}
        self.iou_thr = iou_threshold
        self.max_lost = max_lost
        self.counts = Counter()

    def _iou(self, a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0: return 0.0
        areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
        return interArea / float(areaA + areaB - interArea)

    def update(self, detections):
        assigned_dets = set()
        for tid, tinfo in list(self.tracks.items()):
            best_iou = 0; best_d = -1
            for di, det in enumerate(detections):
                if di in assigned_dets: continue
                if det['cls'] != tinfo['cls']: continue
                i = self._iou(tinfo['box'], det['box'])
                if i > best_iou:
                    best_iou = i; best_d = di
            if best_iou >= self.iou_thr and best_d != -1:
                self.tracks[tid]['box'] = detections[best_d]['box']
                self.tracks[tid]['lost'] = 0
                assigned_dets.add(best_d)
            else:
                self.tracks[tid]['lost'] += 1

        for di, det in enumerate(detections):
            if di in assigned_dets: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {'box': det['box'], 'cls': det['cls'], 'lost': 0}
            self.counts[det['cls']] += 1

        to_delete = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in to_delete: del self.tracks[tid]
        return self.tracks

# ----------------- ClassItem python widget -----------------
class ClassItem(BoxLayout):
    from kivy.properties import StringProperty, NumericProperty
    name_text = StringProperty('')
    count_text = StringProperty('')
    cls_idx = NumericProperty(-1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self._bind_info_btn, 0)

    def _bind_info_btn(self, dt):
        try:
            info_btn = self.ids.info_btn
            # bind to call the app root handler with this widget's cls_idx
            info_btn.unbind(on_release=None)
            info_btn.bind(on_release=lambda inst: App.get_running_app().root.on_info_pressed(self.cls_idx))
        except Exception:
            pass

class VideoDashboard(BoxLayout):
    load_model_text = StringProperty("Load Model")
    load_video_text = StringProperty("Load Video")
    start_stop_text = StringProperty("Start")
    total_text = StringProperty("Total unique organisms: 0")
    processing = BooleanProperty(False)
    counts = DictProperty({})
    fps = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.model_path = ""
        self.source = ""
        self.capture = None
        self.proc_thread = None
        self.stop_flag = threading.Event()
        self.tracker = None
        self.out_writer = None
        self.frame_lock = threading.Lock()
        self.curr_frame = None
        self.last_info_pressed = None
        self.gemini = None
        self.info_helper = None

        # Pi camera fields
        self.use_picamera = False
        self.picam = None

    # -------------- model & file choosers ----------------
    def open_model_chooser(self):
        from kivy.uix.filechooser import FileChooserIconView
        chooser = FileChooserIconView(path='.', filters=['*.pt', '*.pth'])
        popup = self._make_popup("Select model (.pt)", chooser, self._model_selected)
        popup.open()

    def _model_selected(self, chooser, popup):
        if chooser.selection:
            self.model_path = chooser.selection[0]
            self.load_model_text = f"Model: {os.path.basename(self.model_path)}"
            popup.dismiss()
            try:
                self.model = YOLO(self.model_path)
                print("Model loaded:", self.model_path)
            except Exception as e:
                print("Error loading model:", e)

    # ---------- show a choice popup (Camera / Video File / Pi Camera) ----------
    def show_video_options(self):
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(12))
        btn_cam = ModernButton(text="Video File", bg_color=[0.18,0.50,0.30,1], pressed_color=[0.14,0.40,0.24,1])
        btn_file = ModernButton(text="Camera", bg_color=[0.18,0.38,0.60,1], pressed_color=[0.14,0.30,0.48,1])
        content.add_widget(btn_cam)
        content.add_widget(btn_file)

        if PICAMERA2_AVAILABLE:
            btn_picam = ModernButton(text="Pi Camera (libcamera)", bg_color=[0.18,0.45,0.65,1], pressed_color=[0.14,0.36,0.52,1])
            content.add_widget(btn_picam)

        popup = Popup(title="Choose source", content=content, size_hint=(0.45, 0.35), auto_dismiss=True)

        def _choose_camera(instance):
            self.source = '0'
            self.load_video_text = "Video: Camera"
            self.use_picamera = False
            popup.dismiss()

        def _choose_file(instance):
            popup.dismiss()
            self.open_video_chooser()

        def _choose_picam(instance):
            self.source = 'picamera'
            self.load_video_text = "Video: Pi Camera"
            self.use_picamera = True
            popup.dismiss()

        btn_cam.bind(on_release=_choose_camera)
        btn_file.bind(on_release=_choose_file)
        if PICAMERA2_AVAILABLE:
            btn_picam.bind(on_release=_choose_picam)

        popup.open()

    def open_video_chooser(self):
        from kivy.uix.filechooser import FileChooserIconView
        chooser = FileChooserIconView(path='.', filters=['*.mp4','*.avi','*.mov','*.*'])
        popup = self._make_popup("Select video (or choose '0' for webcam)", chooser, self._video_selected)
        popup.open()

    def _video_selected(self, chooser, popup):
        if chooser.selection:
            self.source = chooser.selection[0]
            self.load_video_text = f"PiCamera"
            popup.dismiss()

    def _make_popup(self, title, content_widget, on_select):
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(content_widget)
        btn = Button(text='Select', size_hint_y=None, height='36dp')
        def _on_btn(*a):
            on_select(content_widget, popup)
        btn.bind(on_release=_on_btn)
        layout.add_widget(btn)
        popup = Popup(title=title, content=layout, size_hint=(0.9,0.9))
        return popup

    # --------------- start / stop processing ----------------
    def toggle_processing(self):
        if not self.processing:
            if not self.model:
                print("Load a model first.")
                return
            if not self.source:
                print("Load a video first.")
                return
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        self.stop_flag.clear()
        self.tracker = SimpleTracker(iou_threshold=0.3, max_lost=25)

        # initialize capture based on selection
        if getattr(self, "use_picamera", False) and PICAMERA2_AVAILABLE:
            try:
                self.picam = Picamera2()
                preview_cfg = self.picam.create_preview_configuration(
                    {"main": {"size": [640, 480], "format": "RGB888"}}
                )
                self.picam.configure(preview_cfg)
                self.picam.start()
                self.capture = None
                print("Picamera2 started (640x480 RGB).")
            except Exception as e:
                print("Failed to start Picamera2:", e)
                self.use_picamera = False
                # fallback to v4l2 / cv2
                src = 0 if self.source == '0' else self.source
                self.capture = cv2.VideoCapture(src)
        else:
            src = 0 if self.source == '0' else self.source
            self.capture = cv2.VideoCapture(src)
            if not self.capture.isOpened():
                print("Cannot open source:", self.source)
                # best-effort modprobe bcm2835-v4l2 and retry
                try:
                    import subprocess
                    subprocess.run(["sudo", "modprobe", "bcm2835-v4l2"], check=False)
                    self.capture = cv2.VideoCapture(src)
                except Exception:
                    pass

        # set up writer using width/height
        if getattr(self, "use_picamera", False) and self.picam is not None:
            width, height = 640, 480
        else:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        out_path = "output_dashboard.mp4"
        fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture is not None else 25.0
        try:
            fps = fps or 25.0
        except Exception:
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self.out_writer = cv2.VideoWriter(out_path, fourcc, fps, (int(width), int(height)))
        except Exception as e:
            print("Warning: could not create VideoWriter:", e)
            self.out_writer = None

        self.processing = True
        self.start_stop_text = "Stop"

        self.proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.proc_thread.start()
        Clock.schedule_interval(self._update_image_widget, 1/30.0)

    def stop_processing(self):
        self.stop_flag.set()
        self.processing = False
        self.start_stop_text = "Start"
        if self.proc_thread and self.proc_thread.is_alive():
            self.proc_thread.join(timeout=2.0)
        if getattr(self, "capture", None):
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
        if getattr(self, "out_writer", None):
            try:
                self.out_writer.release()
            except Exception:
                pass
            self.out_writer = None

        # stop picamera if used
        if getattr(self, "picam", None) is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
            self.picam = None
            self.use_picamera = False

    # --------------- processing loop ----------------
    def _processing_loop(self):
        frame_idx = 0
        start_t = time.time()
        while not self.stop_flag.is_set():
            # acquire frame either from Picamera2 or OpenCV
            if getattr(self, "use_picamera", False) and PICAMERA2_AVAILABLE and self.picam is not None:
                try:
                    frame_rgb = self.picam.capture_array()
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    ret = True
                except Exception as e:
                    print("Picamera2 capture error:", e)
                    ret = False
                    frame = None
            else:
                ret, frame = (False, None)
                if self.capture is not None:
                    ret, frame = self.capture.read()

            if not ret or frame is None:
                # end if stream ended or error
                if not self.use_picamera and (self.capture is None or not getattr(self.capture, "isOpened", lambda: True)()):
                    break
                time.sleep(0.01)
                continue

            frame_idx += 1

            # run detection
            try:
                # for speed on Pi, consider downsizing before inference:
                # small = cv2.resize(frame, (416, 416))
                # results = self.model.predict(small, conf=0.25, iou=0.45, verbose=False)
                results = self.model.predict(frame, conf=0.20, iou=0.45, verbose=False)
            except Exception as e:
                print("Model inference error:", e)
                break

            detections = []
            if results:
                res = results[0]
                for box in res.boxes:
                    try:
                        xy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
                        conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
                        detections.append({'box':[x1,y1,x2,y2], 'cls':cls, 'conf':conf})
                    except Exception:
                        continue

            tracks = self.tracker.update(detections)

            vis = frame.copy()
            for det in detections:
                x1,y1,x2,y2 = det['box']
                cls = det['cls']
                conf = det['conf']
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, label, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            counts = self.tracker.counts
            total_unique = sum(counts.values())

            # subtle overlay for saved video output
            box_w = 320
            box_h = 20 * (1 + len(counts))
            overlay = vis.copy()
            cv2.rectangle(overlay, (10,10), (10+box_w, 10+box_h), (0,0,0), -1)
            vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)

            y0 = 30
            cv2.putText(vis, f"Total unique organisms: {total_unique}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 1, cv2.LINE_AA)
            y0 += 26
            for cls_idx, cnt in sorted(counts.items(), key=lambda x:-x[1]):
                name = self.model.names.get(cls_idx, str(cls_idx))
                cv2.putText(vis, f"{name}: {cnt}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
                y0 += 20

            if self.out_writer:
                try:
                    self.out_writer.write(vis)
                except Exception as e:
                    print("Error writing frame:", e)

            with self.frame_lock:
                self.curr_frame = vis.copy()

        elapsed = time.time() - start_t
        print(f"Processing finished, frames: {frame_idx}, elapsed: {elapsed:.1f}s")
        self.processing = False
        self.start_stop_text = "Start"

    # --------------- UI update ---------------
    def _update_image_widget(self, dt):
        img_widget = self.ids.video_image
        frame = None
        with self.frame_lock:
            if self.curr_frame is not None:
                frame = self.curr_frame.copy()
        if frame is None:
            return

        # Fit image widget size
        w = max(2, int(img_widget.width or 320))
        h = max(2, int(img_widget.height or 240))
        frame_resized = cv2.resize(frame, (w, h))
        buf = cv2.flip(frame_resized, 0).tobytes()
        tex = Texture.create(size=(w, h), colorfmt='bgr')
        tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        img_widget.texture = tex

        counts = self.tracker.counts if self.tracker else Counter()
        total_unique = sum(counts.values())
        self.total_text = f"Total unique organisms: {total_unique}"

        box = self.ids.classes_box
        box.clear_widgets()
        for cls_idx, cnt in sorted(counts.items(), key=lambda x:-x[1]):
            name = self.model.names.get(cls_idx, str(cls_idx)) if self.model else str(cls_idx)
            ci = ClassItem()
            ci.name_text = f"{name}"
            ci.count_text = f"{cnt}"
            ci.cls_idx = cls_idx
            box.add_widget(ci)

    # --------------- info pressed handler (Gemini) ---------------
    def on_info_pressed(self, cls_idx):
        self.last_info_pressed = cls_idx
        cls_name = self.model.names.get(cls_idx, str(cls_idx)) if self.model else str(cls_idx)
        print(f"[INFO BUTTON] pressed for class {cls_idx}: {cls_name}")

        if self.gemini is None:
            try:
                self.gemini = GeminiClient()
                self.info_helper = InfoPopupHelper(self, self.gemini)
            except Exception as e:
                content = BoxLayout(orientation='vertical', padding=8, spacing=8)
                content.add_widget(Label(text="Gemini API key not configured.\nSet GOOGLE_API_KEY environment variable."))
                from kivy.uix.button import Button
                btn = Button(text="Close", size_hint_y=None, height=36)
                content.add_widget(btn)
                p = Popup(title="Gemini not configured", content=content, size_hint=(0.6, 0.3))
                btn.bind(on_release=p.dismiss)
                p.open()
                return

        if self.info_helper:
            extra_context = None
            self.info_helper.show_for_class(cls_idx, cls_name, extra_context=extra_context)

    def save_csv(self):
        if not self.tracker:
            print("No counts yet.")
            return
        save_path = "dashboard_counts.csv"
        with open(save_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["class_index","class_name","unique_count"])
            for cls_idx, cnt in sorted(self.tracker.counts.items(), key=lambda x: x[0]):
                writer.writerow([cls_idx, self.model.names.get(cls_idx, str(cls_idx)), cnt])
        print("Saved counts to", save_path)

class YoloDashboardApp(App):
    def build(self):
        Builder.load_string(KV)
        return VideoDashboard()

if __name__ == "__main__":
    YoloDashboardApp().run()