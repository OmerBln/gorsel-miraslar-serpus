# Görsel Miraslar: Osmanlı Serpuşları Sınıflandırma

Bu proje, PyTorch (ResNet101, ConvNeXt-Tiny) ve Streamlit kullanılarak geliştirilmiş, Osmanlı dönemine ait farklı serpuş (başlık) türlerini yapay zeka ile otomatik olarak tanıyan bir görüntü sınıflandırma (Image Classification) uygulamasıdır.

## 🌟 Özellikler
- **Derin Öğrenme Modeli:** Transfer learning kullanılarak eğitilmiş ResNet101 mimarisi.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş, anında sonuç veren hızlı ve etkileşimli web arayüzü.
- **Tarihi Bilgi Entegrasyonu:** Model sınıflandırma yaptıktan sonra, tespit edilen başlık türü hakkında (`serpus_content.json` dosyasından çekilen) kısa tarihi bilgiler sunar.
- **Bilinmeyen Görsel Tespiti:** Sistemin güven (confidence) skoru düşükse (belirlenen eşik değerinin altındaysa), görselin bir Osmanlı başlığı olmadığını anlayarak kullanıcıyı uyarır.

## 🗂 Sınıflandırılan Başlık Türleri (Sınıflar)
Model şu an 5 temel kategoride eğitim almıştır:
1. **Börk** 
2. **Fes** 
3. **Kadın Başlıkları**
4. **Katip** 
5. **Sarma Sarıklı**

---

## 🚀 Kurulum ve Çalıştırma

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/OmerBln/gorsel-miraslar-serpus.git
cd gorsel-miraslar-serpus
```

### 2. Gerekli Kütüphaneleri Yükleyin
Projenin çalışması için gerekli kütüphaneleri kurun:
```bash
pip install -r requirements.txt
```

### 3. Veri Setini Hazırlayın
Projenin ana dizininde bir `data` klasörü oluşturun ve görsellerinizi sınıf isimlerine göre aşağıdaki yapıda yerleştirin:
   ```text
   data/
   ├── train/
   │   ├── Börk/
   │   ├── Fes/
   │   └── ...
   └── validation/
       ├── Börk/
       ├── Fes/
       └── ... 
```
**Veri Seti İndirme Linki Kaggle'a Veriler Yüklendiğinde Paylaşılacak **


### 4. Modeli Kendi Ortamınızda Eğitin (ÖNEMLİ ⚠️)
GitHub dosya boyutu sınırları nedeniyle önceden eğitilmiş model ağırlıkları (`.pth` dosyası) bu depoda bulunmamaktadır.
Arayüzü çalıştırmadan önce `resnet101.py` dosyasını kullanarak modeli kendi ortamınızda eğitmeniz gerekmektedir:

GÜNCELLEME⚠️⚠️ Son eklenen ConvNeXt-Tiny modeli ile %93.06 accuracy elde edilmiştir. İsterseniz `convnext.py` dosyasını kullanarak modeli kendi ortamınızda eğitebilirsiniz.


### 5.Uygulamayı Güncelleyin
Eğitim başarıyla tamamlandığında hangi modeli çalıştırdığınıza bağlı olarak ana dizinde best_model_resnet101_epoch_X_acc_Y.pth veya best_model_convnext_tiny_epoch_X_acc_Y.pth formatında bir dosya oluşacaktır. streamlitv2.py (ResNet için) veya streamlitv3.py (ConvNeXt için) dosyasını açın ve MODEL_PATH değişkenini bu yeni dosyanın adıyla güncelleyin.


### 6. Uygulamayı Başlatın
Tüm adımları tamamladıktan sonra, Streamlit arayüzünü başlatmak için terminale şu komutu yazın:
```bash
streamlit run streamlitv2.py
```
Bu komut, varsayılan web tarayıcınızda otomatik olarak projeyi açacaktır (genellikle `http://localhost:8501`).

---


## 🤝 Katkıda Bulunma
Bu proje tarihi mirasımızı teknoloji ile buluşturmayı amaçlamaktadır. Veri setini genişletmek veya kodlara katkıda bulunmak için Pull Request gönderebilirsiniz.
