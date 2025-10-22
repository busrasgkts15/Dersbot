import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os.path

import subprocess, sys, os, streamlit as st

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="📘 DersBot AI Asistan", page_icon="🤖", layout="centered"
)

# if st.button("🧠 Veritabanını oluştur"):
#     try:
#         result = subprocess.run(
#             [sys.executable, "src/ingest_all.py"],
#             check=True,
#             capture_output=True,
#             text=True,
#             cwd=os.getcwd(),  # ekledik!
#         )
#         st.success("✅ Veritabanı başarıyla oluşturuldu!")
#         st.text(result.stdout)
#     except subprocess.CalledProcessError as e:
#         st.error(f"⚠️ Veritabanı oluşturulamadı! Hata kodu: {e.returncode}")
#         st.code(e.stderr)


# --- Ortam Değişkenleri ---
# Load GOOGLE_API_KEY from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


if not api_key:
    st.error("❌ GOOGLE_API_KEY could not be found. Please check your .env env file.")
    st.stop()

# --- Chroma DB Ayarı (ingest_all.py ile aynı olmalı) ---
SINGLE_DB_PATH = "chroma_db/all_courses_db"


st.markdown(
    """
<style>
/* === GENEL === */
.stApp {
    background-color: #f7f9fc;
    color: #1a1a1a;
    font-family: 'Inter', sans-serif;
}
header, [data-testid="stHeader"], footer { visibility: hidden !important; }

/* === BAŞLIK === */
.main-center-title {
    text-align: center;
    padding-top: 2.5rem; /* Daha az üst boşluk */
    padding-bottom: 1.5rem;
}
.main-center-title .icon {
    font-size: 3em;
    margin-bottom: 0.3rem;
    color: #2563eb;
}
.main-center-title .app-name {
    font-size: 2.4em;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 0;
    margin-bottom: 0.8rem;
}
.main-center-title .app-info {
    font-size: 0.9em;
    color: #f97316;
    background-color: #fff7ed;
    padding: 0.6rem 1.3rem;
    border-radius: 8px;
    border: 1px solid #fed7aa;
    max-width: 600px;
    margin: 1rem auto 2.5rem auto;
    text-align: center;
}

/* === DERS SEÇİMİ === */
div[data-testid="stSelectbox"] {
    max-width: 380px;
    margin: 0 auto 1.2rem auto;
}

/* === INPUT ALANI === */
div[data-testid="stTextInput"] {
    max-width: 760px;
    margin: 0 auto 1rem auto; /* Daha az alt boşluk */
}
div[data-testid="stTextInput"] > div {
    border: none;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    transition: box-shadow 0.3s ease;
    display: flex;
    align-items: center;
    background-color: #ffffff;
}
div[data-testid="stTextInput"] > div:focus-within {
    box-shadow: 0 0 0 2px #2563eb, 0 4px 10px rgba(37, 99, 235, 0.1);
}
div[data-testid="stTextInput"] input {
    flex-grow: 1;
    font-size: 1.05em;
    padding: 1rem 1.2rem;
    color: #1a1a1a !important;
    background-color: transparent;
    border: none !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #9ca3af !important;
}

/* === ÖRNEK SORULAR === */
.example-question-btn-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
}
div[data-testid*="stButton"] > button {
    background-color: #ffffff !important;
    color: #2563eb !important;
    border: 1px solid #dbeafe !important;
    border-radius: 18px !important;
    padding: 0.55em 1.1em !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
div[data-testid*="stButton"] > button:hover {
    background-color: #eff6ff !important;
    border-color: #93c5fd !important;
}

/* === CEVAP BALONU === */
.response-container {
    margin-top: 1.5rem;
    padding: 1.8rem;
    background-color: #ffffff;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
    min-height: 220px; /* sabit yükseklik ile balon boyutları eşitlendi */
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.response-container h4 {
    color: #2563eb;
    font-size: 1.3em;
    margin-bottom: 1rem;
    text-align: left;
}
.response-container p {
    margin: 0;
    line-height: 1.6;
    color: #1f2937;
}

/* === ALERT (HATA) === */
.stAlert {
    background-color: #fff !important;
    color: #1a1a1a !important;
    border-left: 4px solid #f87171 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-radius: 8px;
}

/* === Kod ve Vurgu === */
.stMarkdown code {
    background-color: #eef2ff !important;
    color: #3b82f6 !important;
    padding: 3px 6px;
    border-radius: 6px;
    font-weight: 500;
}
.stCode {
    background-color: #eef2ff !important;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
}
.stCode code {
    color: #1a1a1a !important;
    background-color: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Title Area (Başlık ve Slogan) ---
st.markdown(
    """
<div class="main-center-title">
    <div class="icon">
        📘
    </div>
    <div class="app-name">DersBot AI Asistan</div>
    <div class="app-info">
        ⚠️ **Önemli Bilgilendirme:** DersBot, yüklenmiş akademik notlarınız, sunumlarınız ve kitaplarınız kullanılarak oluşturulmuştur. 
        Yanıtların doğruluğunu her zaman kendi kaynaklarınızdan kontrol etmeniz önerilir.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Ders Seçimi Yönetimi
if "selected_lesson" not in st.session_state:
    st.session_state.selected_lesson = "Sayısal Analiz"

# Ders Seçim Çubuğu, artık görünür etiketle
selected_lesson = st.selectbox(
    "Lütfen bir ders seçiniz:",  # Etiket güncellendi
    options=[
        "Sayısal Analiz",
        "Algoritma Analizi",
        "Mikroişlemciler",
        "İşletim Sistemleri",
    ],
    index=0,
    label_visibility="visible",  # Görünür yapıldı
    key="lesson_selector",  # Anahtarı değiştirdim
)

# 🔄 Ders değiştiğinde input'u sıfırla
if "last_selected_lesson" not in st.session_state:
    st.session_state.last_selected_lesson = selected_lesson

if selected_lesson != st.session_state.last_selected_lesson:
    st.session_state.query = ""  # input alanını temizle
    st.session_state.last_selected_lesson = selected_lesson  # yeni dersi güncelle
    st.rerun()


# --- AI and Database Preparation (Read-Only Control) ---
try:
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = SINGLE_DB_PATH

    # 2. Check Database Directory
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.error(
            f"❌ Database for all courses ({db_path}) **could not be found or is empty**. Please run the 'ingest' code to create it."
        )
        st.stop()

    # 3. Load Database (READ ONLY)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = db.as_retriever()

except Exception as e:
    st.error(f"❌ An unexpected error occurred while loading the database: {e}")
    st.stop()

# --- LLM and Prompt Settings ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2
)

prompt_template = """
Aşağıdaki bağlama göre soruyu yanıtla. Akademik ama sade bir dil kullan.
Eğer bağlamda bilgi yoksa kullanıcıya samimi bir şekilde bunu belirt. Sadece bağlamdaki bilgiye göre yanıt ver.

Bağlam:
{context}

Soru:
{question}

Yanıt:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# --- Query Input (Centralized, Session State for button handling) ---
if "query" not in st.session_state:
    st.session_state.query = ""

# Input field - Görseldeki gibi sadeleştirildi
query_input = st.text_input(
    "",
    placeholder="💭 Şu anda seçili derse göre soru sorun...",
    label_visibility="collapsed",
    value=st.session_state.query,
    key="user_query_input_field",
)

st.markdown(
    """
<style>
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.4rem !important;
    font-size: 1.1em !important;
    color: #111827 !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #9ca3af !important;
}
div[data-testid="stTextInput"] > div:focus-within {
    box-shadow: 0 0 0 3px rgba(37,99,235,0.25), 0 3px 10px rgba(0,0,0,0.05);
    border-color: #2563eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Example Question Buttons ---
# Dynamic examples based on selection for guidance
if selected_lesson == "Sayısal Analiz":
    example_queries = [
        "Kapalı yöntemlerde kök nasıl bulunur?",
        "Newton-Raphson yöntemini açıkla",
        "Hata türleri nelerdir?",
        "İkiye Bölme Yöntemi JAVA Kodu",
    ]
elif selected_lesson == "Algoritma Analizi":
    example_queries = [
        "Decrease And Conquer stratejisini açıkla",
        "Divide And Conquer stratejisini açıkla",
        "Quick Sort ve Merge Sort farkı nedir?",
        "Rus köylüsü çarpımı nedir?",
    ]
elif selected_lesson == "Mikroişlemciler":
    example_queries = [
        "RISC ve CISC mimarileri farkı nedir?",
        "Kesme (Interrupt) nedir?",
        "8051 mimarisinin temel özellikleri",
    ]
elif selected_lesson == "İşletim Sistemleri":
    example_queries = [
        "Konvoy Etkisi Nedir?",
        "SJF ve STCF arasındaki fark nedir ?",
        "MLFQ nedir ?",
    ]
else:
    example_queries = [
        "Bu ders nedir?",
        "Ana konular nelerdir?",
        "Önemli formüller nelerdir?",
    ]

# Örnek Sorular Görseldeki gibi alt alta listeleniyor
st.markdown("<div class='example-question-btn-container'>", unsafe_allow_html=True)
cols = st.columns(len(example_queries))

for i, q in enumerate(example_queries):
    with cols[i]:
        # On button click, update the session state query and rerun
        if st.button(q, key=f"ex_q_{i}", use_container_width=True):
            st.session_state.query = q
            st.session_state.query_submitted = True  # Flag to trigger submission
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# --- Execute Query ---
final_query = query_input

# Check if a button submitted a query
if "query_submitted" in st.session_state and st.session_state.query_submitted:
    final_query = st.session_state.query  # Use the button's query value
    del st.session_state.query_submitted

# If there is a query to process (from text input or button)
if final_query and final_query.strip():
    with st.spinner("Yanıt hazırlanıyor..."):
        try:
            result = qa.invoke({"query": final_query})

            # Display the result in the styled container (Cevap Kartı)
            st.markdown(
                f"""
                <div class='response-container'>
                    <h4>🤖 DersBot Yanıtı</h4>
                    {result['result']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Hata: {e}")
