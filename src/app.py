import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os.path

# --- Ortam Değişkenleri ---
# Load GOOGLE_API_KEY from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY could not be found. Please check your .env file.")
    st.stop()

# --- Chroma DB Ayarı (ingest_all.py ile aynı olmalı) ---
SINGLE_DB_PATH = "../chroma_db/all_courses_db"

# --- Sayfa Ayarları ---
st.set_page_config(page_title="📘 DersBot", page_icon="🤖", layout="centered")

# --- CSS Tasarım (Visual Interface Styling) ---
st.markdown(
    """
<style>
/* 1. Streamlit'in ana kapsayıcısını ve metin rengini ayarla */
.stApp {
    background-color: #ffffff; /* Arka plan beyaz */
    color: #1a1a1a; /* Yazı rengi siyah */
    font-family: 'Inter', sans-serif;
}

/* 2. Streamlit'in varsayılan başlık ve header'larını gizle */
header, [data-testid="stHeader"] { visibility: hidden !important; }
footer { visibility: hidden; }

/* 3. Ana başlık ve logo için minimalist stil (Görseldeki gibi merkezi) */
.main-center-title {
    text-align: center;
    padding-top: 5rem;
    padding-bottom: 2rem;
}
.main-center-title .icon {
    font-size: 3em;
    margin-bottom: 0.5rem;
    color: #ff4b4b; /* Kırmızı yıldız/ikon rengi */
}
.main-center-title .app-name {
    font-size: 2.2em;
    font-weight: 500;
    color: #1a1a1a;
    margin-top: 0;
    margin-bottom: 3rem;
}

/* 4. Soru inputu (Görseldeki uzun input alanı) */
div[data-testid="stTextInput"] {
    max-width: 700px;
    margin: 0 auto 1.5rem auto; /* Üstten ve alttan boşluk ekler */
}
div[data-testid="stTextInput"] > div {
    /* Genel konteyner stilini hafifletiyoruz */
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0; /* İç dolguyu sıfırlıyoruz, input'un kendi padding'ini kullanacağız */
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); /* Hafif gölge */
    transition: box-shadow 0.3s ease;
}
div[data-testid="stTextInput"] > div:focus-within {
    box-shadow: 0 0 0 2px #2563eb; /* Odaklandığında mavi çerçeve */
}

/* Input alanının içine daha iyi padding ve görünüm */
div[data-testid="stTextInput"] input {
    font-size: 1.1em;
    padding: 0.8rem 1rem; /* Daha geniş padding */
    color: #1a1a1a;
    background-color: #ffffff;
}

/* Streamlit'in varsayılan gönder ikonunu gizle (Çünkü st.text_input kullanıyoruz ve görselde input alanı ile bitiyor) */
div[data-testid="stTextInput"] > div > div > svg {
    visibility: hidden;
}


/* 5. Örnek Soru Butonları (Görseldeki gibi alt alta ortalanmış, küçük butonlar) */
.example-question-btn-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1rem;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
div[data-testid*="stButton"] > button {
    background-color: #ffffff !important; /* Arka plan beyaz */
    color: #4a4a4a !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 18px !important;
    padding: 0.4em 1em !important;
    font-size: 0.85em !important;
    font-weight: 400 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}
div[data-testid*="stButton"] > button:hover {
    background-color: #f5f5f5 !important;
    border-color: #ccc !important;
}

/* 6. Yanıt Alanı (Daha belirgin kutu) */
.response-container {
    margin-top: 2.5rem;
    padding: 2rem;
    background-color: #ffffff; /* Yanıt kutusu arka planı beyaz */
    color: #1a1a1a; /* Yanıt kutusu metni siyah */
    border-radius: 12px;
    /* Belirgin gölge */
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(0, 0, 0, 0.05);
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
.response-container h4 {
    color: #2563eb;
    margin-top: 0;
    font-size: 1.2em;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}
/* Streamlit'in md çıktısındaki tüm elementlerin (p, h1, div vb.) rengini ayarlamak için */
.response-container * {
    color: #1a1a1a !important; 
}

/* 7. Koyu Arka Planlı Yazıları (Kod Blokları) Beyaz Temaya Uyarla (YENİ EKLEME) */
.stMarkdown code {
    /* Inline kod (tek tırnak içindeki) */
    background-color: #f5f5f5 !important; /* Açık gri arka plan */
    color: #4a4a4a !important; /* Koyu gri metin */
    padding: 2px 4px;
    border-radius: 4px;
    font-weight: 600;
}
.stCode {
    /* Blok kod (üç tırnak içindeki) */
    background-color: #f8f8f8 !important; /* Çok açık gri arka plan */
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 1rem;
}
.stCode code {
    /* Blok kod metni */
    color: #1a1a1a !important; /* Siyah metin */
    background-color: #f8f8f8 !important; /* Kodu arka planını kod bloğu arka planıyla eşitle */
}


/* 8. Sidebar'ı gizleyelim, görselde yok */
[data-testid="stSidebar"] {
    display: none;
}

/* 9. Streamlit hata kutularını (Traceback, ErrorBox) beyaz temada görünür yap */
.stAlert {
    background-color: #fff !important;
    color: #1a1a1a !important;
    border-left: 4px solid #ff4b4b !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.stAlert pre, .stAlert code {
    color: #000 !important;
    background-color: #f8f8f8 !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# --- Title Area (Mimics the visual interface: Icon + Text) ---
# Sidebar gizlendiği için Ders Seçimi artık merkezde gösterilecek.
# Görseldeki minimalist Streamlit başlığına odaklanıyoruz.
st.markdown(
    """
<div class="main-center-title">
    <div class="icon">
        <!-- Streamlit Ikonuna benzer bir yıldız veya sembol -->
        ★ 
    </div>
    <div class="app-name">DersBot AI Asistan</div>
</div>
""",
    unsafe_allow_html=True,
)

# Ders Seçimi yerine artık merkezi bir bilgi gösterelim (Opsiyonel)
# st.markdown(f"<div style='text-align:center; font-size:1.1em; margin-top:-2rem;'>**{selected_lesson}** Ders Notlarına Dayalı Asistan</div>", unsafe_allow_html=True)
# Ders Seçimini kaldırdığımız için, seçilen dersi bir gizli session state'te tutalım.
if "selected_lesson" not in st.session_state:
    st.session_state.selected_lesson = "Sayısal Analiz"

# Gizli Selectbox ile ders seçimini yönetelim
selected_lesson = st.selectbox(
    "Lütfen hangi dersle ilgili soru soracağınızı seçin (Görsel stili için gizli):",
    options=[
        "Sayısal Analiz",
        "Algoritma Analizi",
        "Mikroişlemciler",
        "İşletim Sistemleri",
    ],
    index=0,
    label_visibility="hidden",
    key="lesson_selector_hidden",
)

# 🔄 Ders değiştiğinde input'u sıfırla
if "last_selected_lesson" not in st.session_state:
    st.session_state.last_selected_lesson = selected_lesson

# Eğer kullanıcı yeni bir ders seçtiyse
if selected_lesson != st.session_state.last_selected_lesson:
    st.session_state.query = ""  # input alanını temizle
    st.session_state.last_selected_lesson = selected_lesson  # yeni dersi güncelle
    st.rerun()
# sayfayı yeniden yükle (boş inputla)


# --- AI and Database Preparation (Read-Only Control) ---
try:
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = SINGLE_DB_PATH

    # 2. Check Database Directory (Prevents writing/re-creation if not found)
    # Check if the path exists AND if it contains files (i.e., is not empty)
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.error(
            f"❌ Database for all courses ({db_path}) **could not be found or is empty**. Please run the 'ingest' code to create it."
        )
        st.stop()

    # 3. Load Database (READ ONLY)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    # The retriever now searches across all courses in the single database
    retriever = db.as_retriever()

except Exception as e:
    st.error(f"❌ An unexpected error occurred while loading the database: {e}")
    st.stop()

# --- LLM and Prompt Settings ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2
)

# CRITICAL: Prompt template ensures answers are based *only* on the provided context.
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
    "Ask a question...",
    placeholder="Ask a question...",  # Placeholder görseldekine benzetildi
    label_visibility="collapsed",
    value=st.session_state.query,
    key="user_query_input_field",
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
        "Önemli formüller nelerdir ?",
        "Önemli formüller nelerdir ?",
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
# Artık ayırıcı çizgi kullanılmıyor
# st.markdown("---")

# --- Execute Query ---
# The logic here handles both direct text input and button clicks
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

            # Display the result in the styled container
            st.markdown(
                f"""
                <div class='response-container'>
                    <h4>🤖 Asistan Yanıtı</h4>
                    {result['result']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Hata: {e}")
