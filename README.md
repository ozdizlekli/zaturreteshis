# Zatürre Teşhis Sistemi (Pneumonia Detection System)

Bu proje, göğüs röntgeni (X-ray) görüntülerini analiz ederek hastalarda zatürre (pnömoni) olup olmadığını tespit eden derin öğrenme tabanlı bir web uygulamasıdır. Model, VGG16 transfer öğrenme (transfer learning) mimarisi kullanılarak eğitilmiş ve Flask framework'ü ile web ortamına taşınmıştır.

## Öne Çıkan Özellikler

* **Yüksek Doğruluklu Teşhis:** VGG16 (ImageNet ağırlıklarıyla) mimarisi kullanılarak yüksek başarı oranına sahip ikili sınıflandırma (Normal vs. Pnömoni) yapar.
* **Kullanıcı Dostu Web Arayüzü:** Flask tabanlı arayüz sayesinde kullanıcılar sisteme kolayca röntgen yükleyip anında sonuç alabilirler.
* **Gelişmiş Model Eğitimi:** Aşırı öğrenmeyi (overfitting) engellemek için veri artırma (Data Augmentation), Dropout katmanları, EarlyStopping ve ReduceLROnPlateau teknikleri kullanılmıştır.

---

## Kullanılan Teknolojiler

* **Geliştirme Dili:** Python
* **Derin Öğrenme / Makine Öğrenmesi:** TensorFlow, Keras (VGG16)
* **Görüntü İşleme:** OpenCV, Pillow, NumPy
* **Web Çerçevesi:** Flask
* **Veri Analizi ve Görselleştirme:** Matplotlib, Pandas, Scikit-learn

---

## Veri Seti ve Model Eğitimi

Modelin eğitimi için göğüs röntgeni veri seti kullanılmıştır. Veri seti boyutu nedeniyle bu repoda bulunmamaktadır. Veri setine Kaggle üzerinden "Chest X-Ray Images (Pneumonia)" adıyla ulaşabilirsiniz.

Eğitim süreci (Untitled.ipynb) içerisinde:
1. Veriler 224x224 boyutuna getirilmiş ve normalize edilmiştir.
2. Eğitim verilerine rastgele döndürme, yakınlaştırma ve yatay çevirme (shear, zoom, horizontal flip) işlemleri uygulanmıştır.
3. Model optimizasyonu için Adam optimizasyon algoritması ve binary_crossentropy kayıp fonksiyonu kullanılmıştır.

---

## Kurulum ve Çalıştırma

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

**1. Repoyu Klonlayın**
```bash
git clone [https://github.com/ozdizlekli/zaturre_teshis_projesi.git](https://github.com/ozdizlekli/zaturre_teshis_projesi.git)
cd zaturre_teshis_projesi
```
2. Sanal Ortam Oluşturun

```bash
python -m venv venv
```

# Windows için:
venv\Scripts\activate
# macOS/Linux için:
source venv/bin/activate
3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```
4. Model Dosyasını Ekleyin
Eğitilmiş model dosyası (best_vgg16_model.h5) GitHub boyut sınırını aştığı için https://drive.google.com/ adresinden indirilip ana dizine (app.py ile aynı klasöre) yerleştirilmelidir.

5. Uygulamayı Başlatın

```bash
python app.py
```


#Proje Dokümantasyonu
Projenin literatür araştırması, teorik altyapısı ve detaylı model mimarisi hakkında bilgi edinmek için repoda bulunan Proje_Raporuzaturre.pdf dosyasını inceleyebilirsiniz.

#Katkıda Bulunma
Bu proje açık kaynaklıdır ve geliştirmeye açıktır. Geliştirme önerileriniz veya karşılaştığınız hatalar için "Issue" açabilir ya da "Pull Request" gönderebilirsiniz.
