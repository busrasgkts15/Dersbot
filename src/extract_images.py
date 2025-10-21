import os
from pdf2image import convert_from_path

# === AYARLAR ===
DATA_PATH = "../data"  # PDF'lerin ana dizini
OUTPUT_DIR = "../data_images"  # Görsellerin kaydedileceği klasör


def extract_images_from_pdf(pdf_path, output_folder):
    """Bir PDF dosyasındaki tüm sayfaları .png olarak kaydeder."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_folder, exist_ok=True)

    print(f"📄 {pdf_name} işleniyor...")

    # PDF'i görüntülere çevir
    try:
        pages = convert_from_path(pdf_path, dpi=200)  # 200 DPI kalite için ideal
    except Exception as e:
        print(f"⚠️ {pdf_name} dönüştürülürken hata oluştu: {e}")
        return []

    image_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.png")
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    print(f"✅ {pdf_name}: {len(image_paths)} sayfa kaydedildi.\n")
    return image_paths


def process_all_pdfs():
    """Tüm ders klasörlerindeki PDF'leri işler."""
    for course_folder in os.listdir(DATA_PATH):
        course_path = os.path.join(DATA_PATH, course_folder)
        if not os.path.isdir(course_path):
            continue

        print(f"🎓 {course_folder.upper()} klasörü işleniyor...")
        course_output = os.path.join(OUTPUT_DIR, course_folder)
        os.makedirs(course_output, exist_ok=True)

        for file in os.listdir(course_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(course_path, file)
                extract_images_from_pdf(pdf_path, course_output)

    print("🎉 Tüm görseller başarıyla çıkarıldı!")


if __name__ == "__main__":
    process_all_pdfs()
