import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import shutil
import zipfile
from ultralytics import YOLO

# --- 1. KONFIGURASI HALAMAN MOBILE FRIENDLY ---
st.set_page_config(
    page_title="AI Mobile Detector",
    page_icon="📱",
    layout="centered", # 'Centered' lebih enak dilihat di HP daripada 'Wide'
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS AGAR SEPERTI APLIKASI HP ---
st.markdown("""
    <style>
        /* Tombol menjadi bulat dan penuh */
        .stButton>button {
            width: 100%;
            border-radius: 25px;
            height: 3.5rem;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        /* Gambar hasil deteksi melengkung (rounded) */
        img {
            border-radius: 12px;
        }
        /* Menghilangkan padding atas yang terlalu lebar */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
        /* Custom Tab agar lebih besar */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt') # Model Small (Cukup ringan tapi pintar)

try:
    model = load_model()
except Exception as e:
    st.error("Gagal load model. Cek koneksi internet.")
    st.stop()

# --- FUNGSI ZIP ---
def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                           os.path.join(folder_path, '..')))

# --- JUDUL APLIKASI ---
st.title("📱 AI Object Scanner")
st.caption("Upload video, AI akan memotong & mendeteksi objek secara otomatis.")

# --- STEP 1: UPLOAD ---
uploaded_video = st.file_uploader("📂 Upload Video (MP4/AVI)", type=['mp4', 'avi'])

if uploaded_video:
    # Simpan sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Tampilkan Video Player (Kecil saja agar hemat tempat)
    with st.expander("▶️ Lihat Video Asli", expanded=False):
        st.video(uploaded_video)
    
    # --- PENGATURAN (Disembunyikan dalam Expander biar rapi) ---
    with st.expander("⚙️ Pengaturan AI & Jumlah Foto", expanded=True):
        st.write("**Berapa foto yang mau diambil?**")
        num_frames_target = st.slider("", 1, 50, 10, key="num_frames")
        
        st.write("**Sensitivitas Deteksi (Akurasi)**")
        conf_threshold = st.slider("", 0.1, 0.9, 0.25, key="conf")
        st.caption("Geser ke kiri = Lebih banyak deteksi (Agresif).")

    # --- TOMBOL AKSI UTAMA ---
    st.write("---")
    extract_btn = st.button("🚀 MULAI SCANNING", type="primary")
    
    if extract_btn:
        # Persiapan Folder
        output_folder = "hasil_deteksi_mobile"
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Logika ambil frame rata
        if total_frames > num_frames_target:
            frame_indices = np.linspace(0, total_frames - 2, num=num_frames_target, dtype=int)
        else:
            frame_indices = np.arange(total_frames)

        # UI Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames_preview = []
        
        # --- LOOP PROCESS ---
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            # AI Process
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Save
            filename = f"objek_{i+1}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, annotated_frame)
            
            # Store for preview
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frames_preview.append(Image.fromarray(frame_rgb))
            
            # Update UI
            progress_bar.progress(int((i + 1) / len(frame_indices) * 100))
            status_text.text(f"⏳ Memproses... {i+1}/{len(frame_indices)}")
            
        cap.release()
        st.toast("✅ Selesai! Foto berhasil disimpan.", icon="🎉")
        
        # --- TAMPILAN HASIL (MENGGUNAKAN TABS AGAR RAPI) ---
        st.write("---")
        tab1, tab2 = st.tabs(["📸 GALERI HASIL", "⬇️ DOWNLOAD"])
        
        with tab1:
            # Tampilkan Grid 2 Kolom (Pas untuk HP)
            cols = st.columns(2) 
            for i, img in enumerate(frames_preview):
                with cols[i % 2]: # Logika ganjil genap untuk 2 kolom
                    st.image(img, use_container_width=True)
                    st.caption(f"Img {i+1}")
        
        with tab2:
            st.success(f"Total {len(frames_preview)} Foto Tersimpan.")
            # Zip Process
            zip_path = "hasil_scan.zip"
            zip_folder(output_folder, zip_path)
            
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="📥 DOWNLOAD SEMUA (ZIP)",
                    data=fp,
                    file_name="hasil_scan_ai.zip",
                    mime="application/zip",
                    type="primary"
                )

else:
    # Tampilan awal jika belum upload
    st.info("👆 Upload video dulu di atas ya!")