# 🎩 Görsel Miraslar: Osmanlı Serpuşları Sınıflandırma

Bu proje, PyTorch (ResNet101) ve Streamlit kullanılarak geliştirilmiş, Osmanlı dönemine ait farklı serpuş (başlık) türlerini yapay zeka ile otomatik olarak tanıyan bir görüntü sınıflandırma (Image Classification) uygulamasıdır.

## 🌟 Özellikler
- **Derin Öğrenme Modeli:** Transfer learning kullanılarak eğitilmiş ResNet101 mimarisi.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş, anında sonuç veren hızlı ve etkileşimli web arayüzü.
- **Tarihi Bilgi Entegrasyonu:** Model sınıflandırma yaptıktan sonra, tespit edilen başlık türü hakkında (`serpus_content.json` dosyasından çekilen) detaylı tarihi bilgiler sunar.
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

### 3. Model Ağırlıklarını İndirin (ÖNEMLİ ⚠️)
Model dosyasının boyutu (yaklaşık 500 MB) GitHub'ın sınırlarını aştığı için bu depoya eklenmemiştir. Uygulamanın çalışması için eğitimli **`.pth`** model dosyanızı (`best_model_resnet101_epoch_21_acc_83.33.pth` veya `best_model_resnet101_epoch_28_acc_77.78.pth`) projenin ana klasörüne (bu klasöre) manuel olarak eklemeniz gerekmektedir.

*(Not: Eğer bir indirme linkiniz varsa, modeli buradan indirin: `[Model Linki Buraya Gelebilir]`)*

### 4. Uygulamayı Başlatın
Tüm adımları tamamladıktan sonra, Streamlit arayüzünü başlatmak için terminale şu komutu yazın:
```bash
streamlit run streamlitv2.py
```
Bu komut, varsayılan web tarayıcınızda otomatik olarak projeyi açacaktır (genellikle `http://localhost:8501`).

---


## 🤝 Katkıda Bulunma
Bu proje tarihi mirasımızı teknoloji ile buluşturmayı amaçlamaktadır. Veri setini genişletmek veya kodlara katkıda bulunmak için Pull Request gönderebilirsiniz.
