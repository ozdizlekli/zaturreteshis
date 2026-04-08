import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ----- Adım 1: Modeli Yükle -----
# Eğitimden sonra kaydettiğin en iyi modeli yüklüyoruz
model_path = 'best_vgg16_model.h5'
try:
    best_model = load_model(model_path)
    print(f"✅ Model '{model_path}' başarıyla yüklendi!")
except Exception as e:
    print(f"Hata: Model yüklenemedi. Dosya yolunu kontrol edin. Hata: {e}")
    best_model = None

# ----- Adım 2: Tahmin Fonksiyonunu Oluştur -----
# Gradio arayüzünden gelen görüntüyü işleyip tahmin yapan ana fonksiyon
def predict_pneumonia(img):
    if best_model is None:
        return "Model yüklenemedi, lütfen bir yöneticinizle iletişime geçin."

    # Görüntüyü modele uygun formata getir
    img_resized = tf.image.resize(img, (224, 224))
    img_normalized = img_resized / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_normalized, axis=0) # Batch boyutu ekle

    # Tahmin yap
    prediction = best_model.predict(img_array)[0][0]

    # Sonucu yorumla
    if prediction > 0.5:
        return f"Tahmin: Pnömoni | Olasılık: %{prediction*100:.2f}"
    else:
        return f"Tahmin: Normal | Olasılık: %{(1-prediction)*100:.2f}"

# ----- Adım 3: Gradio Arayüzünü Oluştur -----
# Arayüz için bir başlık ve açıklama
title = "Akciğer Röntgeni Pnömoni Tespiti Sistemi"
description = "Bir göğüs röntgeni görüntüsü yükleyin ve modelin pnömoni veya normal tahminini alın."

# Arayüz bileşenlerini tanımla
image_input = gr.Image(height=224, width=224)
label_output = gr.Label()

# Arayüzü başlat
if best_model is not None:
    gr.Interface(
        fn=predict_pneumonia,       # Hangi fonksiyonu kullanacağını belirt
        inputs=image_input,         # Girdi olarak resim alacağını belirt
        outputs=label_output,       # Çıktı olarak etiket döndüreceğini belirt
        title=title,                # Arayüz başlığı
        description=description,    # Açıklama metni
        live=False                  # Canlı güncellemeyi kapat (performans için)
    ).launch(share=True)
else:
    print("Model yüklenemediği için Gradio arayüzü başlatılamadı.")