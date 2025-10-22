import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Yol ayarları ---
DATA_ROOT = "../data"
# TEK BİR VERİTABANI DİZİNİ TANIMLIYORUZ
SINGLE_DB_PATH = "../data/chroma_db/all_courses_db"

# --- Embedding modeli ---
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_all_pdfs_from_root(root_folder):
    """Kök klasör altındaki tüm ders klasörlerindeki tüm PDF dosyalarını yükler."""
    all_documents = []

    print(f"📁 PDF'ler taranıyor: {root_folder} altındaki tüm klasörler")

    # Kök klasör (../data) altındaki tüm ders klasörlerini dolaş
    for course_folder in os.listdir(root_folder):
        course_path = os.path.join(root_folder, course_folder)

        # Sadece klasörleri işle
        if os.path.isdir(course_path):
            print(f"\n📂 Ders Klasörü: {course_folder}")

            # Ders klasöründeki tüm PDF'leri yükle
            for file in os.listdir(course_path):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(course_path, file)
                    print(f"  📄 Yükleniyor: {file}")

                    try:
                        loader = PyPDFLoader(pdf_path)
                        documents = loader.load()

                        # Her parçaya hangi derse ait olduğunu belirten metadata ekleyelim
                        for doc in documents:
                            doc.metadata["source_course"] = (
                                course_folder  # Hangi dersten geldiği
                            )
                            doc.metadata["source_file"] = file  # Hangi dosyadan geldiği

                        all_documents.extend(documents)
                    except Exception as e:
                        print(f"  ❌ Hata oluştu ({file}): {e}")

    return all_documents


def main():
    print("📘 Tüm ders notları tek bir veritabanında birleştiriliyor...")

    # 1. Tüm PDF'leri Yükle
    all_docs = load_all_pdfs_from_root(DATA_ROOT)

    if not all_docs:
        print(
            "⚠️ Hiçbir PDF dosyası bulunamadı. Lütfen '../data' altındaki ders klasörlerini kontrol edin."
        )
        return

    print(f"\n📑 Toplam {len(all_docs)} belge sayfası yüklendi.")

    # 2. Metin Parçalarına Ayır
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_chunks = text_splitter.split_documents(all_docs)

    print(f"📚 Toplam {len(all_chunks)} metin parçası oluşturuldu.")

    # 3. Veritabanı Dizinini Oluştur
    os.makedirs(SINGLE_DB_PATH, exist_ok=True)

    # 4. Tek Bir Chroma Veritabanı Oluştur ve Kaydet
    print(f"\n💾 Chroma veritabanı oluşturuluyor ve kaydediliyor: {SINGLE_DB_PATH}")

    db = Chroma.from_documents(
        all_chunks, embedding_function, persist_directory=SINGLE_DB_PATH
    )
    db.persist()  # Kaydetme işlemini tamamla

    print(
        f"\n🎉 Tüm ders notları tek bir veritabanına başarıyla işlendi ve kaydedildi!"
    )


if __name__ == "__main__":
    main()
