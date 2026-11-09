"""Streamlit interface for body posture and finger movement analysis."""
from pathlib import Path
import hashlib
import os
import tempfile
import threading
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.pose_detection import PoseDetector
from src.hand_tracking import HandTracker
from src.face_detection import FaceDetector
from src.movement_analysis import analyze_posture
from src.gesture_analysis import count_extended_fingers, recognize_gesture
from src.visualization import draw_people, draw_hands
from src.video_processing import process_video
from src.webcam_processing import OpenCVCamera, WebcamProcessor

st.set_page_config(page_title="AI Body & Finger Tracking", page_icon="🧍", layout="wide")
st.title("🧍 AI Full Body & Finger Movement Tracking")
st.caption("Pretrained pose and hand landmark models with transparent geometric movement rules.")

@st.cache_resource
def models():
    return PoseDetector(), HandTracker()

@st.cache_resource
def face_detector():
    return FaceDetector()

@st.cache_resource
def inference_lock():
    return threading.RLock()

def analyze_frame(bgr, pose, hand_tracker, temporal=None):
    with inference_lock():
        people = pose.detect(bgr)
        found_hands = hand_tracker.detect(bgr)
    annotated = draw_hands(draw_people(bgr.copy(), people), found_hands)
    body = temporal.update(people) if temporal else (analyze_posture(people[0]) if people else {"posture": "No person detected", "hand_raised": False})
    return annotated, body, found_hands

def show_live_frame(slot, frame_bgr, caption):
    """Update the same Streamlit image element without rebuilding an iframe."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    slot.image(frame_rgb, caption=caption, width="stretch")

def show_model_error(exc):
    st.error(f"Could not initialize the vision models: {exc}")
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 126:
        st.warning("PyTorch is installed, but Windows cannot load one of its DLL dependencies. This happens before the model weight download; reinstall a matching PyTorch and TorchVision pair.")
        st.code(
            r".\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision" + "\n" +
            r".\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu",
            language="powershell",
        )
    elif isinstance(exc, OSError) and getattr(exc, "winerror", None) == 1114:
        st.warning("Windows failed to initialize PyTorch because the installed Visual C++ runtime is too old or incomplete. PyTorch requires the current x64 Visual C++ 2015–2022 runtime.")
        st.markdown("[Download the latest x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). Run it, choose **Install** or **Repair**, then restart the terminal (and Streamlit).")
        st.code(r'.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"', language="powershell")
    else:
        st.info("Check the Python environment and network connection; the pretrained YOLO weights download the first time the model is initialized.")

IMAGE_MODE = "Image Upload Detection"
WEBCAM_MODE = "Real-Time Webcam Detection"
VIDEO_MODE = "Video Upload Detection"
mode = st.sidebar.radio("Choose input", [IMAGE_MODE, WEBCAM_MODE, VIDEO_MODE])
st.sidebar.info("YOLO11n Pose weights download automatically the first time they are used.")
live_detection_enabled = False
if mode == WEBCAM_MODE:
    live_detection_enabled = st.sidebar.checkbox(
        "Enable YOLO, face, and hand detection", value=False, key="webcam_detection_enabled"
    )

if mode != WEBCAM_MODE and st.session_state.get("webcam_running", False):
    st.sidebar.warning("The webcam is still open. Return to Live Detection and press Stop Camera to release it.")
pose = hand_tracker = None
model_error = None
models_needed = mode != WEBCAM_MODE or live_detection_enabled
if models_needed:
    try:
        pose, hand_tracker = models()
    except Exception as exc:
        if mode == WEBCAM_MODE:
            model_error = exc
        else:
            show_model_error(exc)
            st.stop()

if mode == IMAGE_MODE:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = np.array(Image.open(uploaded).convert("RGB"))
            output, body, found = analyze_frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), pose, hand_tracker)
            original_col, annotated_col = st.columns(2)
            original_col.image(image, caption="Original image", width="stretch")
            annotated_col.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Annotated image", width="stretch")
            st.subheader("Frame analysis")
            st.write("**Posture:**", body["posture"])
            st.write("**Hand raised:**", "Yes" if body["hand_raised"] else "No")
            if found:
                for i, hand in enumerate(found, 1):
                    fingers = count_extended_fingers(hand["landmarks"])
                    st.write(f'**{hand["label"]} hand {i}:** {recognize_gesture(hand["landmarks"])}')
                    st.write("Extended fingers:", fingers["count"], "—", ", ".join(fingers["extended"]) or "none")
            else:
                st.write("No hands detected")
            st.caption("A still image describes one instant; walking, running, and jumping require a sequence of frames.")
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")
elif mode == VIDEO_MODE:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotated_video.mp4"

    if uploaded:
        uploaded_bytes = uploaded.getvalue()
        upload_key = hashlib.sha256(uploaded_bytes).hexdigest()
        if st.button("Process video", type="primary"):
            staged_output = output_dir / "annotated_video_pending.mp4"
            video_diagnostics = {}
            try:
                with tempfile.TemporaryDirectory(prefix="body_finger_input_") as folder:
                    source = Path(folder) / ("input" + (Path(uploaded.name).suffix or ".mp4"))
                    source.write_bytes(uploaded_bytes)
                    bar = st.progress(0)
                    status = st.empty()
                    status.info("Processing and encoding an H.264 MP4. The video will appear when encoding finishes.")
                    count = process_video(source, staged_output, pose, hand_tracker, bar.progress,
                                          inference_lock(), diagnostics=video_diagnostics)
                if count == 0:
                    raise RuntimeError("No video frames were processed.")
                if not staged_output.is_file() or staged_output.stat().st_size <= 0:
                    raise RuntimeError("FFmpeg produced no final video file or the file is empty.")
                # Replace the persistent result only after the new file has passed validation.
                os.replace(staged_output, output_path)
                video_diagnostics["path"] = str(output_path.resolve())
                st.session_state.annotated_video_path = str(output_path)
                st.session_state.annotated_video_source = upload_key
                st.session_state.annotated_video_frames = count
                st.session_state.annotated_video_size = output_path.stat().st_size
                st.session_state.annotated_video_diagnostics = video_diagnostics
            except Exception as exc:
                staged_output.unlink(missing_ok=True)
                st.error(f"Video processing or playback preparation failed: {exc}")

    # Keep the latest completed result available after reruns and browser refreshes.
    if output_path.is_file():
        try:
            actual_size = output_path.stat().st_size
            if actual_size <= 0:
                raise RuntimeError("The persistent annotated video is empty.")
            with output_path.open("rb") as video_file:
                video_bytes = video_file.read()
            if len(video_bytes) != actual_size:
                raise RuntimeError("Could not read the complete annotated video file.")

            diagnostics = st.session_state.get("annotated_video_diagnostics")
            with st.expander("Video playback diagnostics", expanded=True):
                st.write("**Output video path:**", str(output_path.resolve()))
                st.write(f"**Output file size:** {actual_size / (1024 * 1024):.2f} MB ({actual_size:,} bytes)")
                if diagnostics:
                    st.write("**OpenCV validation:**", "Passed" if diagnostics["opencv"]["valid"] else "Failed")
                    st.write("**Readable frames:**", diagnostics["opencv"]["readable_frames"])
                    st.write("**Video dimensions:**", f'{diagnostics["ffmpeg"]["width"]} ? {diagnostics["ffmpeg"]["height"]}')
                    st.write("**FPS / duration:**", f'{diagnostics["ffmpeg"]["fps"]:g} / {diagnostics["ffmpeg"]["duration"]:.3f} seconds')
                    st.write("**Codec / pixel format:**", f'{diagnostics["ffmpeg"]["codec"]} / {diagnostics["ffmpeg"]["pixel_format"]}')
                    st.write("**FFmpeg full decode:**", "Passed" if diagnostics["ffmpeg"]["ffmpeg_probe_succeeded"] else "Failed")
                st.write("**Browser MIME type:**", "video/mp4")

            st.subheader("Annotated video")
            try:
                # Render one native player so reruns do not replace competing
                # HTML and Streamlit players for the same video.
                st.video(video_bytes, format="video/mp4")
            except Exception as exc:
                st.error(f"Streamlit's video player could not be created: {exc}")
            frame_count = st.session_state.get("annotated_video_frames")
            if frame_count:
                st.success(f"Processed {frame_count} frames. H.264 output passed OpenCV and FFmpeg validation.")
            st.download_button("Download Annotated Video", video_bytes,
                               file_name="annotated_video.mp4", mime="video/mp4")
            st.caption("Movement labels are geometric estimates smoothed across recent frames.")
        except Exception as exc:
            st.error(f"The annotated video could not be prepared for playback: {exc}")

else:
    st.subheader("Live webcam tracking")
    st.write("This mode reads the camera attached to the computer running Streamlit through OpenCV. It does not use the browser camera prompt.")
    st.info("If access fails, enable camera access for desktop apps in Windows Privacy settings and close other apps that may be using the camera.")
    st.session_state.setdefault("webcam_running", False)
    st.session_state.setdefault("webcam_error", None)
    st.session_state.setdefault("webcam_status", "Camera is stopped.")
    st.session_state.setdefault("webcam_frame_ok", False)
    st.session_state.setdefault("webcam_frame_dimensions", "Not captured")
    st.session_state.setdefault("webcam_face_status", "Not run")
    running = st.session_state.get("webcam_running", False)
    camera_index = st.selectbox("Camera index", [0, 1, 2], index=0, disabled=running)
    if os.name == "nt":
        backend_map = {"DirectShow": cv2.CAP_DSHOW,
                       "Media Foundation": cv2.CAP_MSMF,
                       "OpenCV default": cv2.CAP_ANY}
        default_backend = "DirectShow"
    else:
        backend_map = {"OpenCV default": None}
        default_backend = "OpenCV default"
    backend_name = st.selectbox("Camera backend", list(backend_map),
                                index=list(backend_map).index(default_backend), disabled=running)

    camera = st.session_state.get("webcam_capture")
    camera_config = (camera_index, backend_name)
    camera_changed = camera is None or getattr(camera, "config", None) != camera_config
    if camera_changed:
        if camera is not None:
            camera.stop()
        camera = OpenCVCamera(camera_index, backend_map[backend_name], backend_name)
        camera.config = camera_config
        st.session_state.webcam_capture = camera
        st.session_state.webcam_capture_open = False
        st.session_state.webcam_frame_ok = False
        st.session_state.webcam_frame_dimensions = "Not captured"
        st.session_state.webcam_face_status = "Not run"
        st.session_state.webcam_error = None
        st.session_state.webcam_status = "Camera is stopped."

    live_detector_error = model_error
    if live_detection_enabled and live_detector_error is None:
        try:
            if "webcam_face_detector" not in st.session_state:
                st.session_state.webcam_face_detector = face_detector()
        except Exception as exc:
            live_detector_error = exc
    detectors_ready = live_detection_enabled and live_detector_error is None
    if live_detection_enabled and not detectors_ready:
        st.error(f"AI detectors could not initialize; raw camera preview remains available. {live_detector_error}")
    if detectors_ready and (camera_changed or "webcam_processor" not in st.session_state
                            or st.session_state.webcam_processor is None):
        st.session_state.webcam_processor = WebcamProcessor(
            pose, hand_tracker, inference_lock(), st.session_state.webcam_face_detector
        )
    elif not detectors_ready:
        st.session_state.webcam_processor = None

    def start_webcam():
        """Create or reuse the session capture, then validate its first frame."""
        camera = st.session_state.get("webcam_capture")
        if camera is None:
            camera = OpenCVCamera(camera_index, backend_map[backend_name], backend_name)
            camera.config = camera_config
            st.session_state.webcam_capture = camera
        st.session_state.webcam_error = None
        st.session_state.webcam_status = "Opening camera..."
        st.session_state.webcam_frame_ok = False
        try:
            camera.start()
            st.session_state.webcam_capture_open = camera.is_open
            first_frame = camera.read()
            st.session_state.webcam_processor = (
                WebcamProcessor(pose, hand_tracker, inference_lock(), st.session_state.webcam_face_detector)
                if detectors_ready else None
            )
            st.session_state.webcam_first_frame = first_frame
            st.session_state.webcam_running = True
            st.session_state.webcam_frame_ok = True
            st.session_state.webcam_frame_dimensions = f"{first_frame.shape[1]}x{first_frame.shape[0]}"
            st.session_state.webcam_face_status = "Pending first detector frame"
            st.session_state.webcam_status = "Camera opened and raw frame captured."
        except Exception as exc:
            camera.stop()
            st.session_state.webcam_running = False
            st.session_state.webcam_capture_open = camera.is_open
            st.session_state.webcam_frame_ok = False
            st.session_state.webcam_error = str(exc)
            st.session_state.webcam_status = "Camera startup failed"

    def stop_webcam():
        """Release the device and remove session references to its capture."""
        camera = st.session_state.get("webcam_capture")
        if camera is not None:
            camera.stop()
        st.session_state.webcam_running = False
        st.session_state.webcam_capture_open = False
        st.session_state.webcam_frame_ok = False
        st.session_state.webcam_frame_dimensions = "Not captured"
        st.session_state.webcam_face_status = "Not run"
        st.session_state.webcam_processor = None
        st.session_state.pop("webcam_first_frame", None)
        st.session_state.pop("webcam_capture", None)
        st.session_state.webcam_error = None
        st.session_state.webcam_status = "Camera stopped and released."

    running = st.session_state.get("webcam_running", False)
    start_col, stop_col, _ = st.columns([1, 1, 3])
    with start_col:
        st.button("Start Camera", type="primary", disabled=running,
                  key="opencv_start_camera", on_click=start_webcam)
    with stop_col:
        st.button("Stop Camera", disabled=not running,
                  key="opencv_stop_camera", on_click=stop_webcam)

    camera = st.session_state.get("webcam_capture")
    st.info(st.session_state.get("webcam_status", "Camera is stopped."))
    if st.session_state.get("webcam_error"):
        st.error(f"Webcam error: {st.session_state.webcam_error}")
    with st.expander("Camera diagnostics", expanded=bool(st.session_state.get("webcam_error"))):
        st.write("**Camera index:**", camera.camera_index if camera is not None else camera_index)
        st.write("**Camera backend:**", camera.backend_name if camera is not None else backend_name)
        st.write("**cap.isOpened():**", camera.is_open if camera is not None else False)
        st.write("**Frame capture success:**", st.session_state.get("webcam_frame_ok", False))
        st.write("**Frame dimensions:**", st.session_state.get("webcam_frame_dimensions", "Not captured"))
        st.write("**Face detection:**", st.session_state.get("webcam_face_status", "Not run"))
        if st.session_state.get("webcam_error"):
            st.write("**Camera / processing error:**", st.session_state.webcam_error)

    # Stable placeholders live outside the timed fragment. Each tick replaces
    # only the image data, so Streamlit does not rebuild the preview layout.
    raw_col, preview, analysis = st.columns([1.1, 1.7, 1])
    raw_placeholder = raw_col.empty()
    annotated_placeholder = preview.empty()
    analysis_placeholder = analysis.empty()

    @st.fragment(run_every=0.3)
    def live_camera_panel():
        if not st.session_state.get("webcam_running", False):
            return
        camera = st.session_state.get("webcam_capture")
        if camera is None:
            st.session_state.webcam_running = False
            st.session_state.webcam_capture_open = False
            st.session_state.webcam_error = "Camera session state had no capture object. Press Start Camera to retry."
            st.session_state.webcam_status = "Camera state recovered; camera is stopped."
            st.error(st.session_state.webcam_error)
            return

        first_frame = st.session_state.pop("webcam_first_frame", None)
        try:
            frame = first_frame if first_frame is not None else camera.read()
        except Exception as exc:
            camera.stop()
            st.session_state.webcam_running = False
            st.session_state.webcam_capture_open = camera.is_open
            st.session_state.webcam_frame_ok = False
            st.session_state.webcam_error = str(exc)
            st.session_state.webcam_status = "Frame capture failed; camera was released."
            st.error(f"Webcam error: {exc}")
            return

        st.session_state.webcam_capture_open = camera.is_open
        st.session_state.webcam_frame_ok = True
        st.session_state.webcam_frame_dimensions = f"{frame.shape[1]}x{frame.shape[0]}"
        show_live_frame(raw_placeholder, frame, "Raw webcam frame (before detection)")
        if first_frame is not None:
            return
        if not live_detection_enabled:
            annotated_placeholder.info("Raw camera preview is running. Enable the AI detector checkbox in the sidebar after confirming this image updates.")
            st.session_state.webcam_face_status = "Detection disabled (raw preview mode)"
            return
        if not detectors_ready:
            annotated_placeholder.warning("Raw camera is running; AI detectors are not ready. See the detector error above.")
            return

        try:
            annotated, body, detected_hands, fps, debug = st.session_state.webcam_processor.process_bgr(frame)
            show_live_frame(annotated_placeholder, annotated, "Live annotated webcam")
            face_scores = debug["face_confidences"]
            face_status = ("Face detected" + (f" (confidence {max(face_scores):.2f})" if face_scores else "")) if face_scores else "No face detected"
            st.session_state.webcam_face_status = face_status
            with analysis_placeholder.container():
                st.metric("Processing FPS", f"{fps:.1f}")
                st.caption(f"Faces: {debug['face_count']} ? YOLO people: {debug['person_count']} ? Visible tracked: {debug['tracked_person_count']} ? Active tracks: {debug['active_track_count']} ? IDs: {', '.join(map(str, debug['track_ids'])) or 'none'}")
                st.caption(f"Body keypoints ?0.25: {debug['keypoint_count']}/{debug['keypoint_total']} ? Mean confidence: {debug['keypoint_confidence']:.2f}")
                st.caption(f"MediaPipe hands: {debug['hand_count']} ? Frame: {debug['width']}?{debug['height']} ? Inference: {debug['inference_ms']:.0f} ms")
                st.caption(f"Detector time: YOLO {debug['pose_ms']:.0f} ms ? Hands {debug['hands_ms']:.0f} ms ? Face {debug['face_ms']:.0f} ms ? Shared BGR?RGB {debug['color_conversion_ms']:.1f} ms")
                st.write("**Face result:**", face_status)
                if debug["person_analyses"]:
                    for item in debug["person_analyses"]:
                        st.write(f"**ID {item['track_id']} posture:**", item["posture"])
                        st.write(f"**ID {item['track_id']} movement:**", item["movement"])
                        st.write(f"**ID {item['track_id']} hand raised:**", "Yes" if item["hand_raised"] else "No")
                else:
                    st.write("No tracked person with body keypoints")
                if detected_hands:
                    for hand_index, detected_hand in enumerate(detected_hands, 1):
                        finger_result = count_extended_fingers(detected_hand["landmarks"])
                        confidence = detected_hand.get("confidence")
                        confidence_label = f" ({confidence:.2f})" if confidence is not None else ""
                        st.write(f'**{detected_hand["label"]} hand {hand_index}{confidence_label}:** {recognize_gesture(detected_hand["landmarks"])}')
                        st.write(f'Extended fingers: {finger_result["count"]} ({", ".join(finger_result["extended"]) or "none"})')
                else:
                    st.write("No hands detected")
        except Exception as exc:
            st.session_state.webcam_error = str(exc)
            st.session_state.webcam_face_status = "Detection did not complete"
            st.session_state.webcam_status = "Camera is open; detector processing failed."
            st.error(f"Live detector error (camera remains open): {exc}")

    live_camera_panel()

with st.expander("About the results"):
    st.write("Landmarks come from pretrained models. Posture, finger counts, gestures, and action labels are approximate geometric rules, not learned action classifications.")
    st.write("Dynamic action recognition needs multiple visible frames. Camera motion, unusual poses, and occlusion can change the result.")
