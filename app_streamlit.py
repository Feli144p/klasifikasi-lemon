import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Belajar Klasifikasi Lemon",
	page_icon = ":lemon:"
)

model = joblib.load("model_klasifikasi_lemon.joblib")

st.title(":lemon: Belajar Klasifikasi Lemon")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas lemon")

diameter = st.slider("Diameter", 45.0, 70.0, 55.0)
berat = st.slider("Berat", 65.0, 160.0, 100.0)
tebal_kulit = st.slider("Tebal Kulit", 2.5, 6.5, 4.5)
kadar_gula = st.slider("Kadar Gula", 6.50, 9.00, 7.00)
asal_daerah = st.pills("Asal Daerah", ["Malang", "Medan", "California"], default="California" )
warna = st.pills("Warna", ["Hijau pekat", "Kuning kehijauan", "Kuning cerah"], default="Hijau pekat")
musim_panen = st.pills("Musim Panen", ["Awal", "Akhir", "Puncak"], default="Puncak")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :lemon: oleh **Feliw Sigam**")

















