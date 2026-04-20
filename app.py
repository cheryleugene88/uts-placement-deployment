import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. CONFIG & UI/UX STYLING ---
st.set_page_config(page_title="Placement Intelligence", page_icon="✨", layout="wide")

# Custom CSS: Modern, Clean, Minimalist High-Tech dengan aksen Gold & Black
st.markdown("""
    <style>
    /* Styling tombol menjadi warna Emas (Gold) */
    div.stButton > button:first-child {
        background-color: #D4AF37;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #F3E5AB;
        box-shadow: 0px 4px 15px rgba(212, 175, 55, 0.4);
    }
    /* Warna aksen text header menjadi Emas */
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Mempercantik kotak notifikasi */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    # Pastikan nama file disesuaikan dengan file .pkl yang ter-generate dari script MLflow Anda
    # Gunakan try-except agar Streamlit tidak crash jika file belum dipindahkan
    try:
        # Ganti string di bawah dengan nama file model .pkl Anda yang sebenarnya
        # Contoh: "best_placement_clf_Random_Forest.pkl"
        clf_model = joblib.load("best_placement_clf_Random_Forest.pkl") 
        reg_model = joblib.load("best_salary_reg_Random_Forest_Reg.pkl")
        return clf_model, reg_model
    except Exception as e:
        return None, None

clf_model, reg_model = load_models()

# --- 3. SIDEBAR NAVIGATION & INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Ikon ilustrasi
    st.title("Menu Utama")
    st.markdown("Aplikasi prediksi **Placement Outcomes** dan estimasi **Salary** menggunakan Machine Learning.")
    st.markdown("---")
    st.info("💡 **Tips:** Pastikan Anda mengisi seluruh parameter di form utama untuk mendapatkan prediksi yang akurat.")
    st.markdown("---")
    st.caption("Deployment by: [Nama Anda]")

# --- 4. MAIN INTERFACE ---
st.title("✨ Placement Intelligence Engine")
st.markdown("Masukkan profil kandidat di bawah ini untuk menganalisis probabilitas penempatan kerja dan estimasi gaji.")

if clf_model is None or reg_model is None:
    st.error("⚠️ Model belum ditemukan! Pastikan file `.pkl` berada di folder yang sama dengan `app.py`.")
    st.stop()

# --- 5. INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("📝 Formulir Profil Kandidat")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Akademik Dasar**")
        ssc_p = st.number_input("SSC Percentage (%)", min_value=0.0, max_value=100.0, value=75.0)
        hsc_p = st.number_input("HSC Percentage (%)", min_value=0.0, max_value=100.0, value=75.0)
        degree_p = st.number_input("Degree Percentage (%)", min_value=0.0, max_value=100.0, value=70.0)
        cgpa = st.number_input("CGPA (Skala 10)", min_value=0.0, max_value=10.0, value=7.5)
        
    with col2:
        st.markdown("**Penilaian Skill**")
        entrance_score = st.number_input("Entrance Exam Score", min_value=0, max_value=100, value=80)
        tech_score = st.number_input("Technical Skill Score", min_value=0, max_value=100, value=85)
        soft_score = st.number_input("Soft Skill Score", min_value=0, max_value=100, value=80)
        attendance = st.number_input("Attendance Percentage (%)", min_value=0, max_value=100, value=90)
        
    with col3:
        st.markdown("**Pengalaman & Riwayat**")
        internships = st.number_input("Jumlah Internship", min_value=0, max_value=10, value=1)
        projects = st.number_input("Jumlah Live Projects", min_value=0, max_value=10, value=2)
        work_exp = st.number_input("Work Experience (Bulan)", min_value=0, max_value=120, value=0)
        certs = st.number_input("Jumlah Sertifikasi", min_value=0, max_value=20, value=1)
        backlogs = st.number_input("Jumlah Backlogs (Gagal Matkul)", min_value=0, max_value=10, value=0)
        
    submit_button = st.form_submit_button(label="🚀 Analisis Kandidat")

# --- 6. INFERENCE LOGIC & VISUALIZATION ---
if submit_button:
    # Mengumpulkan input dalam urutan yang tepat sesuai dataset latih
    input_data = pd.DataFrame([{
        "ssc_percentage": ssc_p,
        "hsc_percentage": hsc_p,
        "degree_percentage": degree_p,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_score,
        "technical_skill_score": tech_score,
        "soft_skill_score": soft_score,
        "internship_count": internships,
        "live_projects": projects,
        "work_experience_months": work_exp,
        "certifications": certs,
        "attendance_percentage": attendance,
        "backlogs": backlogs
    }])
    
    st.markdown("---")
    st.subheader("📊 Hasil Analisis")
    
    # Menampilkan loading spinner (menambah kesan high-tech)
    with st.spinner('Memproses data menggunakan model Machine Learning...'):
        # Prediksi Klasifikasi
        placement_pred = clf_model.predict(input_data)[0]
        
        # Ekstrak Probabilitas jika model mendukung (seperti LogisticRegression, RandomForest)
        # Jika menggunakan SVR yang tidak di-set probability=True, ini akan diabaikan
        prob_placement = 1.0
        if hasattr(clf_model, "predict_proba"):
             prob_placement = clf_model.predict_proba(input_data)[0][1]
             
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if placement_pred == 1:
                st.success(f"🎉 **Kandidat Diprediksi: DITERIMA KERJA** (Probabilitas: {prob_placement:.1%})")
                
                # Jika diterima, jalankan model regresi untuk gaji
                salary_pred = reg_model.predict(input_data)[0]
                st.info(f"💰 **Estimasi Gaji (LPA): {salary_pred:.2f} Lakhs Per Annum**")
            else:
                st.error(f"⚠️ **Kandidat Diprediksi: BELUM DITERIMA KERJA** (Probabilitas Penempatan: {prob_placement:.1%})")
                st.warning("Estimasi Gaji tidak tersedia karena kandidat diprediksi belum mendapatkan penempatan.")
                
        # --- DATA VISUALIZATION (Radar Chart) ---
        with col_res2:
            st.markdown("**Komparasi Skill & Akademik**")
            categories = ['SSC', 'HSC', 'Degree', 'CGPA (x10)', 'Tech Skill', 'Soft Skill']
            
            # Normalisasi nilai CGPA agar seimbang di chart (skala 0-100)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                  r=[ssc_p, hsc_p, degree_p, cgpa*10, tech_score, soft_score],
                  theta=categories,
                  fill='toself',
                  name='Kandidat',
                  line_color='#D4AF37'
            ))
            fig.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
              showlegend=False,
              margin=dict(l=20, r=20, t=20, b=20),
              paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)