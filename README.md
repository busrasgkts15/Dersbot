# 🎓 DersBot AI Asistan

**DersBot AI**, kendi eğitim hayatım boyunca derslerde tutmuş olduğum notlarda yer alan bilgileri kullanarak bilgiye hızlı erişim sağlayan
**RAG (Retrieval-Augmented Generation)** tabanlı bir yapay zekâ destekli ders asistanıdır.

Bu proje,Gitbookda oluşturmuş olduğum ders içeriklerini PDF formatında analiz ederek
kullanıcının doğal dilde sorduğu sorulara **not bazlı, sade ve güvenilir** yanıtlar verir.
Modern bir **Streamlit** arayüzüyle sunulur ve tam anlamıyla **LLM + Embedding + Vector DB** entegrasyonu içerir.

---

## 🧠 RAG (Retrieval-Augmented Generation) Mimarisi

DersBot, klasik RAG hattının tüm bileşenlerini uygular:

### 1. 📥 İçerik Yükleme (Content Ingestion)
- `ingest_all.py` dosyası, `data/` klasöründeki tüm PDF’leri otomatik olarak tarar.
- Belgeler `PyPDFLoader` ile okunur.
- Her belgeye, **ders adı (source_course)** ve **dosya ismi (source_file)** metadata olarak eklenir.

### 2. ✂️ Metin Bölme (Text Chunking)
- `RecursiveCharacterTextSplitter` kullanılarak uzun dokümanlar anlamlı parçalara ayrılır.
- **Parametreler:**
  - `chunk_size = 1500`
  - `chunk_overlap = 200`
- Bu ayarlar, bağlam bütünlüğünü koruyarak en doğru bilgi eşleşmesini sağlar.

### 3. 🔡 Embedding (Vektör Dönüştürme)
- Her metin parçası, **HuggingFaceEmbeddings** modeli ile sayısal vektörlere dönüştürülür.
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Avantaj: Hızlı, düşük maliyetli ve Türkçe metinlerde yüksek performans.

### 4. 🗂️ Vektör Veritabanı (Vector Store)
- Vektörler, **Chroma** veritabanında saklanır.
- Tüm dersler tek bir birleşik dizinde tutulur:


### 5. 🔍 Bilgi Getirme (Context Retrieval)
- Kullanıcı sorgusu embedding’e dönüştürülür.
- `db.as_retriever()` fonksiyonu ile benzer içerikler bulunur.
- `RetrievalQA` zinciri üzerinden en alakalı not parçaları seçilir.

### 6. 🧾 Yanıt Üretimi (Response Generation)
- Seçilen bağlam, **Gemini 2.5 Flash** modeliyle analiz edilir.
- Prompt sistemi sayesinde:
- Akademik ama sade bir dil kullanılır.
- “Bağlam” yerine **notlar** formatında cevap oluşturulur.
- Yanıtlar gerçek zamanlı olarak Streamlit arayüzünde gösterilir.

---

## 🖥️ Arayüz Özellikleri

- **Modern ve minimalist tasarım**
- **Renkli placeholder’lar ve yumuşak gölgelendirme**
- **Ders seçimi:** Sayısal Analiz, Algoritma Analizi, Mikroişlemciler, İşletim Sistemleri
- **Dinamik örnek sorular:** Her ders için hazır soru butonları
- **Cevap kutuları:** Sabit boyutlu, gölgeli “baloncuk” görünümü
- **Anlık cevaplama:** Gemini Flash API ile hızlı yanıt üretimi

<img src="https://github.com/user-attachments/assets/03a81b9d-derbot-ui-preview.png" width="750">

## 📚 Veri İşleme Akışı
➡️ `ingest_all.py` bu PDF’leri işler, metinleri parçalara ayırır ve embedding işlemi sonrası  
tek bir Chroma veritabanına kaydeder:  


---

## ⚙️ Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|------------|
| **LLM** | Gemini 2.5 Flash (Google GenAI) |
| **Embeddings** | HuggingFace – `all-MiniLM-L6-v2` |
| **Vector DB** | Chroma |
| **Framework** | LangChain |
| **UI Framework** | Streamlit |
| **PDF İşleme** | PyPDFLoader |
| **Metin Bölme** | RecursiveCharacterTextSplitter |
| **Ortam Değişkeni** | python-dotenv |
| **Programlama Dili** | Python 3.11+ |

---


---

## ⚡ Kurulum ve Çalıştırma

### 1. Depoyu klonla
```bash
git clone https://github.com/busrasgkts15/DersBot.git
cd DersBot
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
````
pip install -r requirements.txt

GOOGLE_API_KEY=your_gemini_api_key

