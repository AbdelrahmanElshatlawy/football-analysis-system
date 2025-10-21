import os
import sys

# --------------------
# Performance & env
# --------------------
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
# Disable Streamlit watcher to avoid double imports during dev
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

# Initialize Streamlit first
import streamlit as st
import tempfile
import cv2
import time

# Import processor with clear error handling for torch
try:
    from video_processor import process_video
    if 'uploaded_path' not in st.session_state:
        st.session_state.uploaded_path = None
    if 'processed_path' not in st.session_state:
        st.session_state.processed_path = None
except RuntimeError as e:
    if "torch" in str(e).lower():
        st.error("Error initializing PyTorch modules. Please restart the application.")
        sys.exit(1)
    raise e
except Exception:
    st.warning("`video_processor` import failed — make sure video_processor.py exists. Processing won't work until fixed.")
    def process_video(*args, **kwargs):
        raise RuntimeError("process_video unavailable - video_processor import failed")

# --- Page config ---
st.set_page_config(page_title="Football Analysis", page_icon="⚽", layout="wide")

# --- Modern CSS (non-overlapping; preserves native DnD) ---
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #071029 100%); color: #e6eef8 }
    .card { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 12px; box-shadow: 0 6px 30px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.04); }
    .muted { color:#9fb7dc; font-size:13px }
    .stButton>button { background: linear-gradient(90deg,#06b6d4,#10b981) !important; color: #042c36; font-weight:700; padding: 8px 18px; border-radius: 10px; box-shadow: 0 8px 18px rgba(6,182,212,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
left_h, right_h = st.columns([0.06, 0.94])
with left_h:
    st.image(r"C:\Users\aelsh\Downloads\connor-coyne-OgqWLzWRSaI-unsplash.jpg", width=56, clamp=True)
with right_h:
    st.markdown("<div style='font-size:32px;font-weight:900'>Football Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Detect players, render trajectories, export tracks — lightweight UI</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# --- Main layout: two columns ---
main_col, side_col = st.columns([0.68, 0.32])

with main_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### Upload Video")
    st.markdown('<div class="muted">Supported: mp4, avi, mov — drag & drop below or click to browse</div>', unsafe_allow_html=True)

    # Native Streamlit uploader (no overlaying HTML around it)
    uploaded_file = st.file_uploader(
        label="Choose a video file",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=False,
        key='video_uploader',
        help='Drag & drop a video file here or click to select',
        label_visibility='visible'
    )

    # If a file is uploaded, save and show player
    if uploaded_file is not None:
        # Save uploaded file to a temp file so st.video can play it from disk
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            tfile.write(uploaded_file.read())
            tfile.flush()
        finally:
            tfile.close()

        # Gather video metadata
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        cols = st.columns([1,1,1,1])
        cols[0].metric("Frames", f"{total_frames}")
        cols[1].metric("FPS", f"{fps}")
        cols[2].metric("Resolution", f"{width}×{height}")
        try:
            cols[3].metric("Size", f"{round(os.path.getsize(tfile.name)/1024/1024,1)} MB")
        except Exception:
            cols[3].metric("Size", "? MB")

        st.markdown("---")
        st.markdown("### Uploaded Video (Preview)")
        # Play the uploaded video file directly (native player with controls)
        try:
            st.video(tfile.name)
        except Exception:
            # fallback to first-frame image preview
            try:
                vidcap = cv2.VideoCapture(tfile.name)
                ok, frame = vidcap.read()
                if ok:
                    h, w = frame.shape[:2]
                    target_w = 720
                    target_h = int(target_w * h / w)
                    frame = cv2.resize(frame, (target_w, target_h))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, caption="First frame preview", use_container_width=True)
                vidcap.release()
            except Exception:
                st.write("Preview not available")

        st.markdown("---")
        st.markdown("### Run Models")
        control_cols = st.columns([1,1])
        with control_cols[0]:
            enable_trajectories = st.checkbox("Enable trajectories", value=False)
        with control_cols[1]:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        # Process button
        if st.button("Process Video"):
            progress_bar = st.progress(0)
            status = st.empty()

            # create output temp file
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_temp.name
            output_temp.close()

            def progress_callback(frame_num):
                if total_frames > 0:
                    p = int((frame_num / float(total_frames)) * 100)
                    progress_bar.progress(min(p, 100))
                    status.info(f"Processing — {p}%")

            try:
                process_video(
                    tfile.name,
                    output_path,
                    enable_trajectories=enable_trajectories,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                # attempt cleanup
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
                try:
                    os.unlink(output_path)
                except Exception:
                    pass
                raise

            progress_bar.progress(100)
            status.success("Processing complete — ready to view")

            # read processed video into memory for display/download
            video_bytes = None
            try:
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
            except Exception as e:
                st.error(f"Failed to read processed video: {e}")

            # remove the source temp file early
            try:
                os.unlink(tfile.name)
            except Exception:
                pass

            if video_bytes:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                st.markdown('### Result')
                # st.video(video_bytes)
                st.download_button("Download processed video", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")
                st.markdown('</div>', unsafe_allow_html=True)

            # cleanup processed file
            try:
                os.unlink(output_path)
            except Exception:
                pass

    else:
        st.info("No video uploaded yet — drop a file to begin")

    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### System info')
    try:
        import platform
        st.write(f"Python: {platform.python_version()}")
        import torch
        st.write(f"PyTorch: {torch.__version__}")
        st.write(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                st.write(f"CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception:
        st.write("Torch not available in this environment")

    st.markdown('---')
    st.markdown('### Tips')
    st.markdown("""
                <div class="muted"><ul style="margin:6px 0 0 18px">
                    <li>Use a modern browser (Chrome/Edge) for best drag-and-drop support.</li>
                    <li>Trim long recordings before uploading to speed processing</li></ul></div>
                """, 
                unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# footer
st.markdown("<div style='margin-top:16px' />", unsafe_allow_html=True)
footer1, footer2 = st.columns([1,1])
with footer1:
    st.markdown('<div class="muted">Made with ❤️ — Football Analysis</div>', unsafe_allow_html=True)
with footer2:
    st.markdown('<div class="muted" style="text-align:right">Version 1.2 — Modern UI</div>', unsafe_allow_html=True)

# End of file


