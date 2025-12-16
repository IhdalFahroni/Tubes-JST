
### **1. Konfigurasi Hiperparameter (*Hyperparameters*)**

Pemilihan hiperparameter didasarkan pada keseimbangan antara efisiensi komputasi dan kemampuan model untuk konvergen (mencapai solusi optimal).

* **Ukuran Citra (`IMG_SIZE = 160`)**
    * **Justifikasi:** Resolusi $160 \times 160$ piksel dipilih sebagai titik tengah yang optimal. Resolusi ini cukup besar untuk mempertahankan fitur visual penting (seperti pola bercak pada daun) namun cukup kecil untuk meminimalkan beban komputasi (VRAM GPU).
    * **Konteks:** Model seperti MobileNetV2 dilatih pada resolusi standar $224 \times 224$. Menguranginya menjadi 160 mempercepat pelatihan sekitar 30-40% dengan penalti akurasi yang minimal pada dataset yang fitur visualnya cukup distingtif seperti penyakit tanaman.

* **Ukuran Batch (`BATCH_SIZE = 16`)**
    * **Justifikasi:** Ukuran batch kecil (16) dipilih karena dua alasan. Pertama, keterbatasan memori (terutama saat melatih model berat seperti VGG16). Kedua, batch kecil memberikan efek *regularisasi* karena estimasi gradien yang lebih bising (*noisy*), yang sering kali membantu model menghindari *local minima* yang tajam dan mencapai generalisasi yang lebih baik.

* **Laju Pembelajaran (`LEARNING_RATE = 0.0001`)**
    * **Justifikasi:** Nilai $10^{-4}$ adalah standar emas untuk *Fine-Tuning*. Karena kita menggunakan model pra-latih (Transfer Learning), bobot awal model sudah sangat baik. Laju pembelajaran yang terlalu besar (misal $0.001$) berisiko "menghancurkan" fitur-fitur berharga yang sudah dipelajari model dari ImageNet. Kita ingin model beradaptasi secara perlahan (*subtle adaptation*).

* **Jumlah Epoch (`EPOCHS = 10`)**
    * **Justifikasi:** Dalam konteks *Transfer Learning* pada dataset yang relatif sederhana (klasifikasi daun dengan fitur jelas), model cenderung konvergen dengan sangat cepat. Melatih lebih dari 10-20 epoch sering kali hanya menghasilkan *overfitting* (menghafal data latih) tanpa peningkatan signifikan pada data validasi.

---

### **2. Strategi Data (*Data Strategy*)**

Keputusan terkait data difokuskan pada validitas input dan penanganan bias.

* **Pembagian Data (`SPLIT_RATIO = 0.2`)**
    * **Justifikasi:** Rasio Pareto (80:20) adalah standar industri. 80% data untuk pelatihan memberikan variasi yang cukup bagi model untuk belajar, sedangkan 20% data validasi cukup representatif untuk mengukur performa model tanpa bias statistik yang signifikan.

* **Augmentasi Data (`rotation`, `shift`, `zoom`, `flip`)**
    * **Justifikasi:** Daun di alam bebas tidak memiliki orientasi kanonik (bisa terbalik, miring, atau terpotong). Augmentasi menyimulasikan variasi fisik ini secara sintetis. Ini memaksa model untuk belajar fitur **invarian** (tahan terhadap perubahan posisi/rotasi), bukan sekadar menghafal piksel pada posisi tertentu.

* **Pembobotan Kelas (`class_weight`)**
    * **Justifikasi:** Dataset penyakit tanaman sering kali tidak seimbang (contoh: sampel penyakit 'Esca' jauh lebih banyak daripada 'Healthy'). Tanpa pembobotan, model akan bias memprediksi kelas mayoritas untuk meminimalkan *error* global secara curang. `class_weight='balanced'` memberikan penalti gradien yang lebih besar jika model salah memprediksi kelas minoritas, memaksa model memperlakukan semua kelas dengan prioritas yang setara.

---

### **3. Pemilihan Arsitektur Model**

Pemilihan model dirancang untuk membandingkan spektrum kompleksitas yang luas.

* **Custom CNN (Baseline)**
    * **Justifikasi:** Berfungsi sebagai *control variable*. Jika model sederhana ini mencapai akurasi tinggi, maka penggunaan model *Deep Learning* yang kompleks mungkin berlebihan (*overkill*). Ini membantu memvalidasi apakah masalah ini benar-benar membutuhkan kapasitas model yang besar.

* **MobileNetV2**
    * **Justifikasi:** Mewakili arsitektur **efisiensi tinggi**. Menggunakan *Depthwise Separable Convolution* untuk meminimalkan parameter. Sangat relevan untuk skenario implementasi di perangkat seluler (*edge computing*) di pertanian.

* **EfficientNetB0**
    * **Justifikasi:** Mewakili **State-of-the-Art (SOTA)** dalam keseimbangan akurasi-efisiensi. Menggunakan *Compound Scaling* yang secara matematis menyeimbangkan kedalaman, lebar, dan resolusi jaringan. Sering kali memberikan performa terbaik dengan parameter minimal.

* **ResNet50V2 & DenseNet121**
    * **Justifikasi:** Mewakili arsitektur **berkapasitas tinggi**. ResNet mengatasi masalah *vanishing gradient* dengan koneksi residual, sementara DenseNet memaksimalkan aliran informasi antar layer. Dipilih untuk melihat apakah fitur yang sangat dalam (*deep features*) diperlukan untuk membedakan penyakit yang mirip.

* **VGG16**
    * **Justifikasi:** Mewakili arsitektur **klasik dan berat**. Meskipun sering kali kalah efisien dari model modern, VGG16 memiliki struktur ekstraksi fitur yang sangat kuat dan sering kali lebih stabil untuk dataset tekstur. Dimasukkan sebagai pembanding historis.

---

### **4. Konfigurasi Pelatihan (*Training Configuration*)**

* **Pengoptimal (*Optimizer*) Adam**
    * **Justifikasi:** Adam (*Adaptive Moment Estimation*) dipilih karena kemampuannya menyesuaikan laju pembelajaran untuk setiap parameter secara individu. Ini sangat efektif untuk masalah dengan data yang jarang (*sparse*) atau *noisy*, dan umumnya konvergen lebih cepat daripada SGD standar tanpa memerlukan penyetelan hiperparameter yang rumit.

* **Fungsi Kerugian (*Loss Function*) Categorical Crossentropy**
    * **Justifikasi:** Karena ini adalah masalah klasifikasi multi-kelas (lebih dari 2 penyakit), *Categorical Crossentropy* adalah fungsi matematis yang tepat untuk mengukur perbedaan (divergensi) antara distribusi probabilitas prediksi model dan label sebenarnya (One-Hot Encoding).

* **Kebijakan Mixed Precision (`float16`)**
    * **Justifikasi:** Menggunakan presisi 16-bit pada GPU modern (NVIDIA Tensor Cores) mempercepat operasi matriks secara drastis dan mengurangi penggunaan memori hingga 50%, memungkinkan penggunaan *batch size* atau model yang lebih besar tanpa mengorbankan akurasi model secara berarti.

---

### **5. Mekanisme Kontrol (*Callbacks*)**

Meskipun tidak secara eksplisit ditulis dalam konfigurasi awal sel 4 (ditambahkan kemudian pada sel pelatihan MobileNet), mekanisme ini krusial:

* **Early Stopping**
    * **Justifikasi:** Mencegah pemborosan sumber daya dan *overfitting*. Jika akurasi validasi tidak membaik setelah beberapa epoch (misal: 3 epoch), pelatihan dihentikan karena model dianggap sudah jenuh atau mulai menghafal data.

* **ReduceLROnPlateau**
    * **Justifikasi:** Strategi "penghalusan". Ketika penurunan *loss* mulai melambat (stagnan), laju pembelajaran dikurangi (misal: dibagi 2 atau 5). Ini memungkinkan model untuk mengambil langkah yang lebih kecil dan hati-hati dalam ruang gradien, sering kali membantu model keluar dari *saddle points* dan mencapai akurasi yang sedikit lebih tinggi.