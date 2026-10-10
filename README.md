# Arabica × Youreye Hackathon Projesi 

Bu proje, hackathon kapsamında istenen iki temel görüntü işleme görevini (Kafe içi kişi/cinsiyet analizi ve Şehir girişi araç/plaka analizi) çözmek amacıyla geliştirilmiş, **uçtan uca çalışan bir yapay zeka ve bilgisayarlı görü** sistemidir.

---

##  Kullandığım Model ve Kütüphanelerin Kısa Açıklaması

Proje mimarisinde hız ve yüksek doğruluğu bir araya getiren "State-of-the-Art" (SOTA) modeller tercih edilmiştir:

1. **YOLO11m (Ultralytics):** Her iki görevde de temel obje tespiti (insan ve araç) için kullanıldı. Yüksek FPS değerlerine çıkabilmesi ve küçük objelerdeki hassasiyeti sebebiyle tercih edildi.
2. **ByteTrack:** YOLO'nun tespit ettiği kişilerin ve araçların kareler (frame) arasında kaybolmadan takip edilmesi, sayım kamerasından geçene kadar aynı ID ile izlenmesi için entegre edildi.
3. **OpenAI CLIP (clip-vit-base-patch16):** Görev 1'de cinsiyet tespiti için kullanıldı. Eski tarz özel eğitilmiş sınıflandırıcılar yerine **Zero-Shot Classification** yöntemiyle çok daha esnek ve yüksek doğruluklu bir sonuç elde edildi. Kendi yazdığımız **Çoklu Çerçeve Oylama (GenderVoter)** sistemiyle, kişinin yürüdüğü süre boyunca alınan çoklu fotoğraflarının CLIP skorları harmanlandı.
4. **YOLO11s-License-Plate-Detection:** Görev 2'de araçların üzerindeki plaka koordinatlarını bulmak için Hugging Face üzerinden alınan, Roboflow veri setleriyle fine-tune edilmiş özel bir model kullanıldı.
5. **EasyOCR (PyTorch/CUDA):** Görüntüden metin çıkarmak için GPU destekli olarak çalıştırıldı. Sadece alfanümerik (harf ve rakam) kısıtlaması ve *Beam Search* algoritmasıyla doğruluk artırıldı.
6. **OpenCV & NumPy:** Görüntü işleme, plaka kesitlerinin (crop) kontrast/keskinlik iyileştirmesi (CLAHE ve Unsharp Masking), ekrana şeffaf bilgi panelleri (UI) ve yön oklarının çizilmesi için yoğun olarak kullanıldı.

---

##  Teknik Özet (Mimari)

Sistem tamamen asenkron okuma yapabilen `ThreadedVideoReader` altyapısı üzerine kurulmuştur. 
* **Görev 1:** Vücut merkezinden (center) veya ayaklardan (bottom) referans alarak vektörel hesaplamayla (Cross Product) sanal bir çizgi üzerinden geçenleri Giriş/Çıkış olarak sayar. Tespit edilen kişilerin resimleri kırpılarak anlık olarak cinsiyet oylamasına sokulur.
* **Görev 2:** Şehre giren araçların tespiti sonrası OCR'a girmeden önce, küçük araçlar elenerek işlem yükü azaltılır. Kalan araç plakaları OpenCV ile okunabilirlik açısından filtrelerden geçirilip EasyOCR'a beslenir. Oylama sistemi, plakanın en çok tekrar eden ve en yüksek güven skoruna sahip halini nihai sonuç olarak seçer. Tüm veriler CSV, JSON ve HTML (Dashboard) olarak anlık dışa aktarılır.

---

##  Bilinen Hatalar, Kısıtlar ve Geliştirme Önerileri

### Kısıtlar & Bilinen Hatalar:
* **Aşırı Kalabalık ve Kapanma (Occlusion):** Kamera açısının yatay olmasından ötürü, insanların veya araçların üst üste binmesi (birinin diğerini kapatması) durumunda takip (tracking) kayıpları yaşanabilmektedir.
* **Açı ve Işık Yansımaları:** Görev 2'deki plaka okuma işleminde, çok dar açılardan veya doğrudan güneş ışığı parlamasından dolayı OCR'ın "O/0" veya "I/1" gibi harf karmaşaları yaşadığı durumlar nadiren görülmektedir.
* **Donanım İhtiyacı:** Gerçek zamanlı analiz ve yüksek FPS için (özellikle OCR tarafında) CUDA destekli sağlam bir GPU'ya (örn: RTX 4060) ihtiyaç duymaktadır.

### Geliştirme Önerileri (Gelecek Vizyonu):
1. **Çoklu Kamera Eşleşmesi (Re-ID):** İlerleyen aşamalarda kafenin farklı yerlerindeki kameralar birbirine bağlanıp (Person Re-Identification), giren bir kişinin içeride ne kadar süre vakit geçirdiği analiz edilebilir.
2. **Kamera Açısı Optimizasyonu:** Kameraların "Kuş Bakışı" (Bird's-eye view) açısına yaklaştırılması, obje örtüşmesi (occlusion) problemini tamamen çözecek ve sayım doğruluğunu %99'un üzerine çıkaracaktır.
3. **LLM Destekli Plaka Düzeltme:** Gelecekte, yabancı veya okuması zor plakaların eksik harflerini tamamlamak için OCR sonuçları hafif bir Dil Modeline (LLM) beslenerek mantıksal tahmin gücü eklenebilir.

---
**Emeği Geçenler:** 
Proje geliştirilirken yapay zekanın sunduğu vizyoner algoritmalar ile gerçek dünyanın kaotik verileri arasında köprü kurulmuştur.
