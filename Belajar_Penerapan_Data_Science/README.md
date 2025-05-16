# Proyek Akhir: Menyelesaikan Permasalahan Jaya Institut!

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis

Tingginya dropout dari keseluruhan pelajar.

### Cakupan Proyek

Membuat model untuk memprediksi pelajar yang memungkinkan dropout menggunakan klasifikasi

### Persiapan

Sumber data: [Dicoding Jaya Institut!](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv)

Setup environment:

- Jalankan docker

```
docker run -d -p 3000:3000 metabase/metabase

```
- Login menggunakan email:root@mail dan password:root123 atau mengakses link publik dibawah

## Business Dashboard

Dashboard ini berfokus terhadap kriteria nilai pelajar.

Link [Dashboard Metabase](http://localhost:3000/public/dashboard/43c55b78-b0b3-4180-9158-37ee277c4154)

## Menjalankan Sistem Machine Learning

- Akses [streamlit app](https://numan-bpds-2.streamlit.app/)

- Mengisi data pelajar

- Tekan predict untuk prediksi

## Conclusion
- 30 Persen dari total keseluruhan mengalami Dropout
- Perempuan memilki tingkat kelulusan lebih tinggi dari Laki-laki
- Ekonomi bukan permasalahan utama pelajar, dikarenakan mayoritas tidak memiliki hutang dan tidak memiliki tunggakan
- Namun, mayoritas pelajar yang dropout adalah yang tidak mendapat beasiswa
- Dari hasil yang didapat, model cukup baik memprediksi kelas Dropout sebesar 0.69 untuk decision tree dan 0.79 untuk random forest.
- Fitur seperti nilai semester (Curricular units 1 & 2) dan nilai pendaftaran (Application rank) memiliki peran penting dalam prediksi status siswa.

 | Fitur Importance                   | Number    |
 |:-----------------------------------|:----------|
 | Curricular_units_2nd_sem_approved  |  0.108982 |
 |    Curricular_units_2nd_sem_grade  |  0.101730 |
 | Curricular_units_1st_sem_approved  |  0.075298 |
 |    Curricular_units_1st_sem_grade  |  0.072926 |
 |                   Admission_grade  |  0.054590 |


### Rekomendasi Action Items (Optional)

Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.

Peningkatan Akademik:
- Berdasarkan tingkat feature importance, nilai per semester merupakan fitur yang penting dalam prediksi, sehingga berhubungan dengan akademik. Aksi yang dapat dilakukan adalah memberikan perhatian khusus pada siswa dengan nilai akademik yang rendah melalui program bimbingan belajar atau konseling dapat membantu meningkatkan performa dan mengurangi risiko dropout

Permasalahan financial:
- Walaupun kebanyakan pelajar tidak memiliki hutang maupun tunggakan, namun mayoritas pelajar yang DO merupakan pelajar yang tidak memiliki beasiswa, aksi yang dapat dilakukan adalah menyediakan opsi pengajuan peringanan pembayaran, seperti mencicil, pengajuan beasiswa pendidikan dan lainnya.