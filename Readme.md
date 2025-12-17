---

### **1. Konfigurasi Lingkungan & Variabel Global**

#### **A. Perangkat Keras (Hardware Settings)**

* **`mixed_precision.set_global_policy('mixed_float16')`**
* **Justifikasi:** Mengubah komputasi dari float32 (32-bit) menjadi campuran float16 dan float32. Ini dipilih untuk **menghemat VRAM GPU hingga 50%** dan mempercepat waktu *training* pada GPU NVIDIA modern (yang memiliki Tensor Cores), tanpa mengurangi akurasi model secara signifikan.


* **`tf.config.experimental.set_memory_growth(gpu, True)`**
* **Justifikasi:** Mencegah TensorFlow mengalokasikan *seluruh* memori GPU sejak awal (yang bisa membuat komputer *hang*). Ini memaksa alokasi memori secara dinamis sesuai kebutuhan proses.



#### **B. Hiperparameter (Hyperparameters)**

* **`IMG_SIZE = 160`**
* **Justifikasi:** Resolusi 160 \times 160 dipilih sebagai kompromi antara **detail visual dan beban komputasi**. Meskipun standar ImageNet adalah 224 \times 224, resolusi 160 mengurangi jumlah piksel yang harus diproses sekitar 50%, mempercepat pelatihan secara drastis. Penyakit daun (bercak/tekstur) biasanya masih dapat diidentifikasi dengan jelas pada resolusi ini.


* **`BATCH_SIZE = 16`**
* **Justifikasi:** Ukuran *batch* kecil dipilih karena dua alasan:
1. **Memori:** Memungkinkan pelatihan model berat (seperti VGG16) tanpa *Out of Memory*.
2. **Generalisasi:** *Batch* kecil menghasilkan estimasi gradien yang lebih *noisy*, yang bertindak sebagai regularisasi implisit, membantu model keluar dari *local minima* dan mencapai generalisasi yang lebih baik.




* **`EPOCHS = 10`**
* **Justifikasi:** Karena menggunakan **Transfer Learning**, model *pre-trained* sudah memiliki fitur ekstraktor yang sangat baik. Konvergensi (pembelajaran) pada dataset baru biasanya terjadi sangat cepat (di bawah 10 epoch). Melatih lebih lama berisiko *overfitting* (menghafal data latih).


* **`LEARNING_RATE = 0.0001` (1e-4)**
* **Justifikasi:** Nilai konservatif (kecil) dipilih. Karena kita melakukan *fine-tuning* pada model yang sudah dilatih, kita tidak ingin mengubah bobot *pre-trained* secara drastis (*catastrophic forgetting*). Laju ini memastikan penyesuaian bobot terjadi secara halus.



---

### **2. Preprocessing & Augmentasi Data**

- **`ImageDataGenerator` (Augmentasi On-the-fly)**
- **Justifikasi:** Dipilih agar variasi data dibuat secara _real-time_ saat training, bukan disimpan statis di hardisk. Ini menghemat penyimpanan dan memberikan variasi yang hampir tak terbatas.

- **Parameter: `rotation=40`, `shift`, `shear`, `zoom`, `horizontal_flip**`
- **Justifikasi:** Daun di alam tidak memiliki orientasi baku (bisa miring, terbalik, terpotong, atau diambil dari jarak berbeda). Augmentasi ini memaksa model untuk belajar fitur **invarian** (mengenali penyakit terlepas dari posisi/rotasi daun), bukan menghafal piksel di posisi tertentu.

- **`class_weight='balanced'`**
- **Justifikasi:** Mengatasi **ketidakseimbangan dataset**. Jika jumlah foto "Sehat" jauh lebih banyak dari "Sakit", model cenderung bias menebak "Sehat". _Class Weight_ memberikan penalti _loss_ lebih besar jika model salah menebak kelas minoritas, memaksa model berlaku adil pada semua kelas.

---

### **3. Arsitektur Model: Custom CNN**

Bagian ini adalah _baseline_ (model dasar) yang dirancang dari nol untuk memvalidasi apakah arsitektur sederhana sudah cukup.

- **1. Layer Konvolusi (`Conv2D`)**
- **Filter (32, 64, 128):** Jumlah filter meningkat secara bertahap (_pyramid architecture_). Layer awal (32) menangkap fitur sederhana (garis, tepi), layer tengah (64) menangkap bentuk (lingkaran bercak), dan layer akhir (128) menangkap fitur kompleks (pola penyakit spesifik).
- **Kernel Size (3, 3):** Ukuran standar yang efisien untuk menangkap detail lokal tanpa parameter berlebih.
- **Padding='same':** Memastikan dimensi output gambar sama dengan input, mencegah informasi di pinggir gambar hilang.
- **Activation='relu':** Memperkenalkan non-linearitas, memungkinkan model mempelajari pola kompleks dan mempercepat konvergensi dibanding Sigmoid/Tanh.

- **2. `BatchNormalization` (setelah setiap Conv)**
- **Justifikasi:** Menormalkan output dari layer sebelumnya agar memiliki mean 0 dan variansi 1. Ini menstabilkan proses pelatihan, mengurangi masalah _internal covariate shift_, dan mempercepat belajar.

- **3. `MaxPooling2D(2, 2)**`
- **Justifikasi:** Melakukan _downsampling_ (mengurangi dimensi spasial setengahnya). Ini mengurangi jumlah parameter yang harus dihitung dan membuat model lebih tahan terhadap pergeseran kecil (_translation invariance_).

- **4. `GlobalAveragePooling2D` (GAP)**
- **Justifikasi:** Dipilih menggantikan `Flatten`. `Flatten` akan mengubah peta fitur 3D menjadi vektor 1D raksasa yang membutuhkan jutaan parameter di layer Dense berikutnya (rentan _overfitting_). GAP merata-rata setiap peta fitur, mengurangi dimensi secara drastis namun tetap mempertahankan informasi spasial global. Ini membuat model jauh lebih ringan.

- **5. `Dense(256)` dengan `kernel_regularizer=l2(0.001)**`
- **Justifikasi:** Layer _fully connected_ untuk penalaran tingkat tinggi. Penambahan **L2 Regularization** memberikan penalti pada bobot yang terlalu besar, mencegah model terlalu bergantung pada satu fitur saja (mencegah _overfitting_).

- **6. `Dropout(0.5)**`
- **Justifikasi:** Teknik regularisasi agresif. Mematikan 50% neuron secara acak selama pelatihan. Ini memaksa jaringan untuk membangun jalur fitur yang berlebihan (_redundant_), sehingga model menjadi lebih _robust_ (tangguh) dan tidak sekadar menghafal.

- **7. `Dense(num_classes, activation='softmax')**`
- **Justifikasi:** Layer output wajib untuk klasifikasi multi-kelas. `Softmax` mengubah output mentah menjadi distribusi probabilitas yang totalnya 1.0 (100%).

---

### **4. Arsitektur Model: Transfer Learning**

Pemilihan model _pre-trained_ didasarkan pada karakteristik unik masing-masing:

1. **MobileNetV2:** Dipilih karena **efisiensi**. Menggunakan _Depthwise Separable Convolution_. Sangat cocok untuk simulasi jika model nantinya akan dideploy di HP petani (Edge AI).
2. **ResNet50V2:** Dipilih karena kemampuan **Deep Learning**. Menggunakan _Residual Connections_ (skip connection) yang mengatasi masalah _vanishing gradient_, memungkinkan jaringan yang sangat dalam belajar tanpa degradasi.
3. **VGG16:** Dipilih sebagai **pembanding klasik**. Arsitektur ini sederhana namun sangat berat parameternya. Bagus untuk mengekstrak tekstur, namun biasanya lambat.
4. **EfficientNetB0:** Dipilih sebagai **State-of-the-Art (SOTA)**. Menggunakan _Compound Scaling_ yang menyeimbangkan kedalaman, lebar, dan resolusi secara optimal. Biasanya memberikan akurasi tertinggi dengan parameter minimal.
5. **DenseNet121:** Dipilih karena **Feature Reuse**. Setiap layer terhubung ke layer lainnya. Sangat bagus untuk dataset kecil karena aliran gradien sangat lancar (memaksimalkan informasi dari setiap gambar).

---

### **5. Proses Pelatihan (Training Loop)**

- **Optimizer: `Adam**`
- **Justifikasi:** Adam (_Adaptive Moment Estimation_) dipilih dibanding SGD karena ia menyesuaikan laju pembelajaran (_learning rate_) untuk setiap parameter secara individu. Ini membuatnya konvergen lebih cepat dan lebih sedikit membutuhkan penyetelan manual.

- **Loss Function: `categorical_crossentropy**`
- **Justifikasi:** Fungsi kerugian standar untuk masalah klasifikasi multi-kelas (>2 kelas) dengan label _one-hot encoded_.

- **`tf.keras.backend.clear_session()` & `gc.collect()**`
- **Justifikasi:** Karena proses ini melatih 6 model secara berurutan dalam satu _loop_, memori RAM/GPU akan cepat penuh. Perintah ini secara eksplisit menghapus model lama dari memori sebelum memuat model baru, mencegah _crash_ (Out of Memory).
