# Capstone-Project-Module-3-Customer-Lifetime-Value-Prediction-

### **Sumber Data**
- [Klik disini (G-Drive)](https://drive.google.com/drive/folders/1RJ5g9FNQ3JNEE4Tyl5KgxJodGniYw8S9) 
- [Atau Klik disini (Kaggle)](https://www.kaggle.com/datasets/pankajjsh06/ibm-watson-marketing-customer-value-data)
### **Daftar Isi**

1. Business Problem Understanding
2. Data Understanding
3. Data Preprocessing
4. Cross Validation
5. Hyperparameter Tuning
6. Modeling
7. Conclusion
8. Recommendation


### **1. Business Problem Understanding**
**Context**

Perusahaan A adalah sebuah perusahaan asal Amerika Serikat yang bergerak di bidang asuransi. Dalam dunia bisnis, kemampuan sebuah perusahaan untuk membuat kliennya "betah" dan loyal terhadap perusahaan itu sangatlah krusial. Klien yang loyal adalah salah satu sumber *profit* yang stabil bagi perusahaan A. Maka dari itu, perusahaan A perlu mengetahui seberapa "berharga" seorang klien dari perusahaan A. Untuk mengukur tingkat "harga" klien, ada sebuah metrik pengukuran yang dinamakan sebagai *__customer lifetime value (CLV)__*. 

*__Customer lifetime value (CLV)__* adalah ukuran seberapa "berharga"nya seorang klien terhadap suatu perusahaan. Ada juga yang berpendapat bahwa *CLV* adalah nilai masa kini dari arus uang (*cash flow*) di masa depan yang dimiliki oleh seorang pelanggan selama hubungannya dengan perusahaan. Secara umum, *customer lifetime value* dapat memberi informasi kepada perusahaan jumlah *revenue* dan *profit* yang dihasilkan dan dapat dihasilkan oleh seorang klien, baik di masa sekarang, maupun masa depan. *Customer lifetime *value* sangatlah penting bagi perusahaan Apabila mereka ingin menargetkan pemasaran yang lebih efektif terhadap klien dengan *value* yang berbeda-beda.

**Problem Statement**

Salah satu tantangan terbesar bagi perusahaan A adalah menentukan nilai atau *value* seorang calon klien agar perusahaan A dapat menentukan pendekatan atau teknik pemasaran yang cocok digunakan untuk klien tersebut.

Dengan variasi aset dan status sosial klien yang beragam, menentukan *CLV* pelanggan dengan tepat untuk dapat tetap kompetitif dan memastikan bahwa mereka mendapat layanan yang memuaskan sangatlah penting.

**Goals**

Berdasarkan permasalahan tersebut, perusahaan A tentu perlu memiliki 'tool' yang dapat memprediksi serta membantu klien mereka (dalam hal ini klien mereka) untuk dapat **menentukan *customer lifetime value* pelanggan berdasarkan fitur-fitur tertentu dengan akurat**. Adanya perbedaan pada berbagai fitur yang dimiliki klien, seperti jenis kendaraan, jumlah pendapatan, dan tingkat edukasi dapat menambah keakuratan prediksi nilai klien tersebut.

Bagi perusahaan A, *prediction tool* yang dapat memberikan prediksi *CLV* secara akurat tentu dapat meningkatkan jumlah klien. Dengan kata lain, semakin banyak klien berarti dapat meningkatkan *revenue* perusahaan, dalam konteks ini didapat dari premi asuransi klien.

**Analytic Approach**

Jadi, yang perlu kita lakukan adalah menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada, yang membedakan satu klien dengan yang lainnya. 

Selanjutnya, kita akan membangun suatu model regresi yang akan membantu perusahaan untuk dapat menyediakan algoritma prediksi *CLV* yang baru masuk dalam daftar calon klien perusahaan A, yang mana akan berguna untuk perusahaan dalam menentukan segmentasi pelanggan.

**Evaluation Metrics**

Metrik evaluasi yang akan digunakan adalah RMSE, MAE, dan MAPE, di mana RMSE adalah nilai rata-rata akar kuadrat dari *error*, MAE adalah rata-rata nilai absolut dari *error*, dan MAPE adalah rata-rata persentase *error* yang dihasilkan oleh model regresi. Semakin kecil nilai RMSE, MAE, dan MAPE yang dihasilkan, berarti model semakin akurat dalam memprediksi *CLV* sesuai dengan limitasi fitur yang digunakan. 

Selain itu, kita juga bisa menggunakan nilai *R-squared* atau *adj. R-squared* jika model yang nanti terpilih sebagai final model adalah model linear. Nilai *R-squared* digunakan untuk mengetahui seberapa baik model dapat merepresentasikan varians keseluruhan data. Semakin mendekati 1, maka semakin *fit* atau cocok pula modelnya terhadap data observasi. Namun, metrik ini tidak valid untuk model non-linear.

### **2. Data Understanding**

- Dataset merupakan data perusahaan asuransi mobil di Amerika pada tahun 2019.
- Setiap baris data merepresentasikan informasi terkait aset, status sosial, dan informasi lain dari seorang klien.

**Features Information**

| **Feature** | **Data Type** | **Description** |
| --- | --- | --- |
| Vehicle Class | Object | Class of Customer Vehicle |
| Coverage | Object | Type of Insurance Policies |
| Renew Offer Type | Object | Type of Renewal Offers |
| Employment Status | Object | Customer Employment Status |
| Marital Status | Object | Customer Marital Status |
| Education | Object | Customer Education Level |
| Number of Policies | Float | Number of Policies Customer Currently Owns |
| Monthly Premium Auto | Float | Amount of customers' monthly insurance payments (in US$)|
| Total Claim Amount | Float | Cumulative Amount of Claims Since Policy Inception|
| Income | Float | Customer Income (in US$)|
| Customer Lifetime Value | Float | Customer Lifetime Value |

[Feature Information Source](https://www.kaggle.com/code/juancarlosventosa/models-to-improve-customer-retention/notebook) 
<br>

Setelah melakukan proses *cross validation*, *randomized search*, dan *hyperparameter tuning*, model terbaik untuk memprediksi nilai *CLV* klien pada *dataset* ini adalah __*Random Forest Regressor*__.

__*Random Forest Regressor*__ adalah sebuah model atau algoritma *machine learning* untuk memprediksi suatu nilai numerikal. Model ini bekerja dengan membuat beberapa pohon keputusan (atau model *decision tree*) dan menggabungkan hasil dari setiap pohon untuk membuat prediksi akhir. Setiap pohon dibuat secara acak dari dataset. Setiap pohon membuat prediksi sendiri dan hasil dari semua pohon digabungkan menjadi prediksi akhir. Gabungan ini biasanya dilakukan dengan mengambil rata-rata dari hasil prediksi dari setiap pohon.

__*Random Forest Regressor*__ memiliki beberapa keuntungan, seperti tingkat stabilitas model yang tinggi dan kemampuan untuk menangani kasus dimana ada fitur yang memiliki pengaruh besar pada target dan juga meminimalisir *overfitting*. Namun, metode ini membutuhkan waktu proses yang lebih lama dibandingkan dengan metode lainnya dan juga membutuhkan lebih banyak memori karena banyaknya pohon yang dibangun. Disini saya mengatasi masalah ini dengan membatasi jumlah pohon yang dibuat didalam parameter n_estimators.

### **7. Conclusion**
Dari hasil *cross validation*, *hyperparameter tuning*, dan *randomized search*, model terbaik adalah __random forest__ dengan ketentuan parameter berikut:

- n_estimators = 187
- min_samples_split = 4
- min_samples_leaf = 3
- max_features = auto
- max_depth = 10

Dengan nilai rata-rata MAPE sebesar 11.69%

---
Berdasarkan pemodelan yang sudah dilakukan, fitur '*Number of Policies*' dan '*Monthly Premium Auto*' menjadi fitur yang paling berpengaruh terhadap '*Customer Lifetime Value*'.

 Metrik evaluasi yang digunakan pada model adalah nilai MAPE. Jika ditinjau dari nilai MAPE yang dihasilkan oleh model setelah dilakukan *hyperparameter tuning*, yaitu sebesar 12.77%, kita dapat menyimpulkan bahwa bila nanti model yang kita buat ini digunakan untuk memperkirakan *CLV* klien baru di perusahaan A pada rentang nilai seperti yang dilatih terhadap model (minimal = 1898.01, maksimal = 58753.88), maka perkiraan *CLV*nya rata-rata akan meleset kurang lebih sebesar 12.77% dari nilai *CLV* seharusnya. 
 
 Tetapi, tidak menutup kemungkinan juga prediksinya meleset lebih jauh karena *bias* yang dihasilkan model masih cukup tinggi bila dilihat dari visualisasi antara *CLV* aktual dan prediksi. *Bias* yang dihasilkan oleh model ini dikarenakan oleh terbatasnya fitur pada *dataset* yang bisa merepresentasikan aspek informasi klien, seperti kelas kendaraan, jenis asuransi, status pekerjaan, status pernikahan, tingkat edukasi, dan lain-lain.

 Model ini tentu masih dapat ditingkatkan performanya agar dapat menghasilkan prediksi yang lebih baik lagi. Namun, kita dapat melakukan A/B testing terhadap model yang sudah dibuat pada project ini untuk mengetahui tingkat efektifitas penggunaan model terhadap perkiraan nilai *CLV*. Nantinya, dari hasil A/B testing, kita bisa mendapatkan *insight* lainnya terkait perihal yang bisa dan harus diperbaiki pada model.  
 
---
Mengapa model kita memprediksi dengan kurang akurat? Karena model ini memiliki beberapa limitasi, diantaranya:
1. Jumlah fitur yang sedikit. Karena *dataset* yang didapatkan hanya berisi 10 fitur (10 kolom), maka tentu saja hasil prediksi hanya akan belajar dari fitur-fitur yang sedikit itu, sehingga model menjadi "kurang pintar". Ternyata, walaupun kita sudah menggunakan *polynomial feature*, namun hal itu tidak dapat meningkatkan performa model.
2. Jumlah data (dapat dihitung dari jumlah baris pada dataset) yang sangat sedikit. Kita hanya memiliki 5669 data diawal, sehingga model hanya dapat belajar dari rentang data-data tersebut. Apabila model disuruh prediksi data lain yang berada diluar dari rentang *dataset* yang kita miliki, maka hasil prediksi menjadi kurang akurat dan kurang bisa dipercaya.
3. Masih ada *outliers*. Di satu sisi, apabila kita menghilangkan seluruh *outliers*, maka kita akan kehilangan lebih banyak data dari data yang jumlahnya sudah sangat sedikit. Di sisi lainnya, *outliers* dapat mengurangi performa model. Jadi, setiap keputusan yang diambil memang ada kelebihan dan kekurangannya.

### **8. Recommendation**

### Untuk bisnis.

Dikarenakan nilai *error* yang cukup besar dari nilai *CLV* aktual dengan *CLV* prediksi, maka hasil prediksi model ini sebaiknya tidak secara mentah-mentah dipercaya oleh perusahaan A untuk memprediksi *CLV* kliennya. Hal ini disebabkan oleh limitasi yang telah dijelaskan diatas. Namun, hasil prediksi untuk nilai *CLV* yang rendah masih cukup akurat. Perusahaan A masih dapat memercayai hasil prediksi model apabila nilai *CLV* masih sekitar dibawah 8000 (berdasarkan visual dari *scatterplot*).

### Untuk performa model.

Perusahaan A dapat memperbaiki performa model melalui beberapa cara, diantaranya:

- Menambah jumlah fitur klien. Hal ini dapat mengurangi risiko model *underfitting*. Namun, tentu saja fitur yang ditambahkan tidak boleh sembarangan. Apabila perusahaan A menambah terlalu banyak fitur, maka model dapat cenderung *overfitting*. Tambahkan fitur-fitur yang berkaitan/berkorelasi kuat dengan *CLV* seseorang, misalkan tingkat retensi klien (semakin lama klien menggunakan produk/jasa perusahaan A, maka semakin tinggi pula nilai *CLV* klien itu), frekuensi pembelian klien (semakin sering klien melakukan pembelian, semakin tinggi nilai *CLV* klien itu), dan demografi klien (segmen klien yang berbeda mungkin memiliki perilaku pembelian (*purchasing behavior*) yang berbeda. Hal ini dapat mempengaruhi *CLV* mereka).

- Menambah jumlah data klien. Hal ini dapat membuat model menjadi semakin "pintar" karena model akan belajar dari data yang lebih heterogen/lebih beragam. Hal ini juga dapat mengurangi kemungkinan *overfitting*.

- Meningkatkan kualitas data. Ada aturan terkenal dalam dunia *machine learning* yaito GIGO (*Garbage In, Garbage Out*). Tidak peduli seberapa canggih dan akurat model yang dipakai, apabila data yang diolah berkualitas rendah, maka hasilnya akan berkualitas rendah juga.
