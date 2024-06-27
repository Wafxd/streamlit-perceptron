import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def main():
    st.image("images/perc.png", width=300)
    
    with st.sidebar :
        page = option_menu ("Pilih Halaman", ["Home", "Data Understanding","Preprocessing", "Model", "Evaluasi","Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        show_understanding()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Model":
        show_model()
    elif page == "Evaluasi":
        show_evaluasi()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Klasifikasi Lokasi TB dengan menggunakan Metode Perceptron")

    # Explain what is Decision Tree
    st.header("Apa itu Perceptron?")
    st.write("Perceptron pada Jaringan Syaraf Tiruan (Neural Network) termasuk kedalam salah satu bentuk Jaringan Syaraf (Neural Network) yang sederhana. Perceptron biasanya digunakan untuk mengklasifikasikan suatu tipe pola tertentu yang sering dikenal dengan istilah pemisahan secara linear. Pada dasarnya perceptron pada Jaringan Syaraf Tiruan (Neural Network) dengan satu lapisan memiliki bobot yang bisa diatur dan suatu nilai ambang. Algoritma yang digunakan oleh aturan perceptron ini akan mengatur parameter-parameter bebasnya melalui proses pembelajaran. Fungsi aktivasi dibuat sedemikian rupa sehingga terjadi pembatasan antara daerah positif dan daerah negative")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data dan klasifikasi dengan menggunakan metode Perceptron.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan diambil dari dosen mata kuliah Kecerdasan Komputasional yang berisi informasi terkait TBC.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Klasifikasi Perceptron")
    st.write("1. **Data Understanding atau Pemahaman Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan**")
    st.write("4. **Evaluasi Model**")
    st.write("5. **Implementasi**")

def show_understanding():
    st.title("Data Understanding")
    data = pd.read_csv("Data TB 987 record.csv")
    
    st.header("Metadata dari data TB")
    st.dataframe(data)
    
    col1, col2 = st.columns(2,vertical_alignment='bottom')
    
    with col1 :
        st.write("Jumlah Data : ", len(data.axes[0]))
    
    with col2 :
        st.write(f"Terdapat {len(data['LOKASI ANATOMI (target/output)'].unique())} Label Kelas, yaitu : {data['LOKASI ANATOMI (target/output)'].unique()}")

    st.markdown("---")
    
    st.header("Tipe Data & Missing Value")
    
    r2col1, r2col2 = st.columns(2,vertical_alignment='bottom')
    
    with r2col1 :
        st.write("Tipe Data")
        st.write(data.dtypes)
    
    with r2col2 :
        st.write("Missing Value")
        st.write(data.isnull().sum())
    
    st.markdown("---")
    
    st.header("Persebaran atribut yang teridentifikasi terdapat missing value")    
        
    tcm = len(data[data['HASIL TCM'] == 'Tidak dilakukan'])
    toraks = len(data[data['FOTO TORAKS'] == 'Tidak dilakukan'])
    hiv = len(data[data['STATUS HIV'] == 'Tidak diketahui'])
    diabet = len(data[data['RIWAYAT DIABETES'] == 'Tidak diketahui'])
    
    # Labels and values for the histogram
    labels = ['HASIL TCM', 'FOTO TORAKS', 'STATUS HIV', 'RIWAYAT DIABETES']
    values = [tcm, toraks, hiv, diabet]
    
    # Create the histogram
    plt.figure(figsize=(7, 3))
    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    
    # Add titles and labels
    plt.title('Jumlah Data yang Tidak Dilakukan/Tidak Diketahui')
    plt.xlabel('Kategori')
    plt.ylabel('Jumlah')
    
    # Display the values on top of the bars
    for i, value in enumerate(values):
        plt.text(i, value + 5, str(value), ha='center')
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    
    st.markdown("---")
    
    st.header("Hasil TCM Berdasarkan Lokasi Anatomi")
    
    kategori_tcm = ['Tidak dilakukan','Rif Sensitif', 'Negatif', 'Rif resisten']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    for kategori in kategori_tcm:
        st.subheader(f"Hasil TCM: '{kategori}'")
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['HASIL TCM'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            st.write(f"{lokasi}: {jumlah}")
    
    kategori_tcm = ['Tidak dilakukan', 'Rif Sensitif', 'Negatif', 'Rif resisten']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    # Calculate counts for each combination of 'HASIL TCM' and 'LOKASI ANATOMI'
    counts = {}
    for kategori in kategori_tcm:
        counts[kategori] = []
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['HASIL TCM'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            counts[kategori].append(jumlah)

    # Plotting
    x = np.arange(len(lokasi_anatomi))  # label locations
    width = 0.2  # width of the bars

    fig, ax = plt.subplots(figsize=(7, 3))

    # Create bars for each 'HASIL TCM' category
    for i, kategori in enumerate(kategori_tcm):
        ax.bar(x + i*width, counts[kategori], width, label=kategori)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Lokasi Anatomi')
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Data Berdasarkan Hasil TCM dan Lokasi Anatomi')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(lokasi_anatomi)
    ax.legend()

    # Display the counts on top of the bars
    for i, kategori in enumerate(kategori_tcm):
        for j, count in enumerate(counts[kategori]):
            ax.text(x[j] + i*width, count + 1, str(count), ha='center')

    fig.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.markdown("---")
    
    
    kategori_foto_toraks = ['Tidak dilakukan', 'Positif', 'Negatif']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    st.header("Hasil Foto Toraks Berdasarkan Lokasi Anatomi")

    for kategori in kategori_foto_toraks:
        st.subheader(f"Hasil Foto Toraks: '{kategori}'")
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['FOTO TORAKS'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            st.write(f"{lokasi}: {jumlah}")
            
    kategori_foto_toraks = ['Tidak dilakukan', 'Positif', 'Negatif']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    # Menghitung jumlah untuk setiap kombinasi 'FOTO TORAKS' dan 'LOKASI ANATOMI'
    counts = {}
    for kategori in kategori_foto_toraks:
        counts[kategori] = []
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['FOTO TORAKS'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            counts[kategori].append(jumlah)

    # Plotting
    x = np.arange(len(lokasi_anatomi))  # label locations
    width = 0.2  # width of the bars

    fig, ax = plt.subplots(figsize=(7, 3))

    # Membuat bar untuk setiap kategori 'FOTO TORAKS'
    for i, kategori in enumerate(kategori_foto_toraks):
        ax.bar(x + i * width, counts[kategori], width, label=kategori)

    # Menambahkan beberapa teks untuk label, judul, dan label khusus sumbu x, dll.
    ax.set_xlabel('Lokasi Anatomi')
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Data Berdasarkan Hasil Foto Toraks dan Lokasi Anatomi')
    ax.set_xticks(x + width)
    ax.set_xticklabels(lokasi_anatomi)
    ax.legend()

    # Menampilkan jumlah di atas bar
    for i, kategori in enumerate(kategori_foto_toraks):
        for j, count in enumerate(counts[kategori]):
            ax.text(x[j] + i * width, count + 1, str(count), ha='center')

    fig.tight_layout()

    # Tampilkan plot dalam Streamlit
    st.pyplot(fig)
    
    st.markdown("---")
    
    kategori_status_hiv = ['Tidak diketahui', 'Positif', 'Negatif']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    st.header("Hasil Status HIV Berdasarkan Lokasi Anatomi")

    # Menghitung jumlah untuk setiap kombinasi 'STATUS HIV' dan 'LOKASI ANATOMI'
    for kategori in kategori_status_hiv:
        st.subheader(f"Hasil Status HIV: '{kategori}'")
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['STATUS HIV'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            st.write(f"{lokasi}: {jumlah}")
            
    counts = {}
    for kategori in kategori_status_hiv:
        counts[kategori] = []
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['STATUS HIV'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            counts[kategori].append(jumlah)

    # Plotting dengan Streamlit
    fig, ax = plt.subplots(figsize=(7, 3))

    # Membuat bar untuk setiap kategori 'STATUS HIV'
    for i, kategori in enumerate(kategori_status_hiv):
        ax.bar(x + i * width, counts[kategori], width, label=kategori)

    # Menambahkan beberapa teks untuk label, judul, dan label khusus sumbu x, dll.
    ax.set_xlabel('Lokasi Anatomi')
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Data Berdasarkan Status HIV dan Lokasi Anatomi')
    ax.set_xticks(x + width)
    ax.set_xticklabels(lokasi_anatomi)
    ax.legend()

    # Menampilkan jumlah di atas bar
    for i, kategori in enumerate(kategori_status_hiv):
        for j, count in enumerate(counts[kategori]):
            ax.text(x[j] + i * width, count + 1, str(count), ha='center')

    fig.tight_layout()

    # Tampilkan plot dalam Streamlit
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("Jumlah Data Berdasarkan Riwayat Diabetes dan Lokasi Anatomi")

    kategori_riwayat_diabetes = ['Tidak diketahui', 'Ya', 'Tidak']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    # Menghitung jumlah untuk setiap kombinasi 'RIWAYAT DIABETES' dan 'LOKASI ANATOMI'
    for kategori in kategori_riwayat_diabetes:
        st.subheader(f"Hasil Riwayat Diabetes: '{kategori}'")
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['RIWAYAT DIABETES'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            st.write(f"{lokasi}: {jumlah}")
            
    kategori_riwayat_diabetes = ['Tidak diketahui', 'Ya', 'Tidak']
    lokasi_anatomi = ['Paru', 'Ekstra paru']

    # Calculate counts for each combination of 'RIWAYAT DIABETES' and 'LOKASI ANATOMI'
    counts = {}
    for kategori in kategori_riwayat_diabetes:
        counts[kategori] = []
        for lokasi in lokasi_anatomi:
            jumlah = len(data[(data['RIWAYAT DIABETES'] == kategori) & (data['LOKASI ANATOMI (target/output)'] == lokasi)])
            counts[kategori].append(jumlah)

    # Plotting with Matplotlib
    x = np.arange(len(lokasi_anatomi))  # label locations
    width = 0.2  # width of the bars

    fig, ax = plt.subplots(figsize=(7, 3))

    # Create bars for each 'RIWAYAT DIABETES' category
    for i, kategori in enumerate(kategori_riwayat_diabetes):
        ax.bar(x + i*width, counts[kategori], width, label=kategori)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Lokasi Anatomi')
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Data Berdasarkan Riwayat Diabetes dan Lokasi Anatomi')
    ax.set_xticks(x + width)
    ax.set_xticklabels(lokasi_anatomi)
    ax.legend()

    # Display the counts on top of the bars
    for i, kategori in enumerate(kategori_riwayat_diabetes):
        for j, count in enumerate(counts[kategori]):
            ax.text(x[j] + i*width, count + 1, str(count), ha='center')

    fig.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)
    
    
    
def show_preprocessing():
    st.title("Preprocessing Data")
    data = pd.read_csv("Data TB 987 record.csv")
    
    st.header("Pengisian Data")
    st.write("Mengisi data yang dianggap sebagai missing value")
    data['FOTO TORAKS'] = data['FOTO TORAKS'].replace('Tidak dilakukan', 'Positif')
    data['STATUS HIV'] = data['STATUS HIV'].replace('Tidak diketahui', 'Negatif')
    data['RIWAYAT DIABETES'] = data['RIWAYAT DIABETES'].replace('Tidak diketahui', 'Tidak')
    data['HASIL TCM'] = data['HASIL TCM'].replace('Tidak dilakukan', 'Rif Sensitif')
    st.dataframe(data)

    # --------------- Drop attribute -----------------
    st.markdown("---")
    
    
    st.header("Merubah data kategori menjadi angka")
    st.write("### Data sebelum diubah menjadi angka")
    col1, col2, col3 = st.columns(3,vertical_alignment='bottom')
    
    with col1 :
        st.write("JENIS KELAMIN:", data['JENIS KELAMIN'].unique())
    
    with col2 :
        st.write("FOTO TORAKS:", data['FOTO TORAKS'].unique())
        
    with col3 :    
        st.write("STATUS HIV:", data['STATUS HIV'].unique())
        
       
    col1, col2, col3 = st.columns(3,vertical_alignment='top')
     
        
    with col1 :
        st.write("RIWAYAT DIABETES:", data['RIWAYAT DIABETES'].unique())
    
    with col2 :
        st.write("HASIL TCM:", data['HASIL TCM'].unique())
    
    with col3 :    
        st.write("LOKASI ANATOMI (target/output):", data['LOKASI ANATOMI (target/output)'].unique())
    

    data['JENIS KELAMIN'] = data['JENIS KELAMIN'].map({'P': 0, 'L': 1})
    data['FOTO TORAKS'] = data['FOTO TORAKS'].map({'Negatif': 0, 'Positif': 1})
    data['STATUS HIV'] = data['STATUS HIV'].map({'Negatif': 0, 'Positif': 1})
    data['RIWAYAT DIABETES'] = data['RIWAYAT DIABETES'].map({'Tidak': 0, 'Ya': 1})
    data['HASIL TCM'] = data['HASIL TCM'].map({'Negatif': 1, 'Rif Sensitif': 0, 'Rif resisten': 2})
    data['LOKASI ANATOMI (target/output)'] = data['LOKASI ANATOMI (target/output)'].map({'Paru': 0, 'Ekstra paru': 1})
    
    st.write("### Data sesudah diubah menjadi angka")
    
    col1, col2, col3 = st.columns(3,vertical_alignment='bottom')
    
    with col1 :
        st.write("JENIS KELAMIN:", data['JENIS KELAMIN'].unique())
    
    with col2 :
        st.write("FOTO TORAKS:", data['FOTO TORAKS'].unique())
        
    with col3 :    
        st.write("STATUS HIV:", data['STATUS HIV'].unique())
        
       
    col1, col2, col3 = st.columns(3,vertical_alignment='top')
     
        
    with col1 :
        st.write("RIWAYAT DIABETES:", data['RIWAYAT DIABETES'].unique())
    
    with col2 :
        st.write("HASIL TCM:", data['HASIL TCM'].unique())
    
    with col3 :    
        st.write("LOKASI ANATOMI (target/output):", data['LOKASI ANATOMI (target/output)'].unique())

    st.markdown("---")
    
    st.header("Drop data yang tidak diperlukan")
    st.write("Drop fitur kecamatan")
    data = data.drop(["KECAMATAN"], axis=1)
    
    st.write(data.head())
    
    st.markdown("---")
    
    st.header("Normalisasi Data dengan Min Max Scalar")
    
    X = data.drop(columns='LOKASI ANATOMI (target/output)')
    y = data['LOKASI ANATOMI (target/output)']
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    st.dataframe(X_scaled)
    
    
    st.session_state['preprocessed_data'] = X_scaled
    st.session_state['target'] = y

def show_model():
    st.title("Testing Model")
    
    if 'preprocessed_data' in st.session_state and 'target' in st.session_state:
        X_scaled = st.session_state['preprocessed_data']
        y = st.session_state['target']
        combined_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        
        st.header("Menggunakan Data yang Sudah di Preprocessing")
        st.dataframe(combined_data)
        
        st.markdown("---")
        
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, train_size=0.8, shuffle=False)
        st.header("Membagi Dataset menjadi Data Training dan Data Testing")
        
        trained = pd.concat([X_train, y_train], axis=1)
        st.write("### Data Training 80%")
        st.dataframe(trained)
        st.write("Jumlah Data : ", len(trained.axes[0]))
        
        
        testing = pd.concat([X_test, y_test], axis=1)
        st.write("### Data Testing 20%")
        st.dataframe(testing)
        st.write("Jumlah Data : ", len(testing.axes[0]))
        
        st.markdown("---")
        
        st.write("### Memasukkan Data Training ke dalam model Perceptron")
        
        st.write("Model Perceptron yang dibentuk menggunakan fungsi aktifasi Undak Biner")
        st.latex(r'y = \begin{cases} 0 & \text{jika } x \leq 0 \\ 1 & \text{jika } x > 0 \end{cases}')
        
        
        st.markdown("---")
        
        
        st.header("Menampilkan Hasil Modelling Perceptron")
        
        clf_perceptron = Perceptron(max_iter=1000)

        clf_perceptron.fit(X_train, y_train)

        y_pred_perceptron = clf_perceptron.predict(X_test)

        df_pred_perceptron = pd.DataFrame(y_pred_perceptron, columns=["Perceptron"])
        df_test = pd.DataFrame(y_test).reset_index(drop=True)

        df_pred_combined = pd.concat([df_pred_perceptron, df_test], axis=1)
        df_pred_combined.columns = ["Perceptron", "Actual Class"]

        class_mapping = {0: "Paru", 1: "Ekstra Paru"}
        df_pred_combined["Perceptron"] = df_pred_combined["Perceptron"].map(class_mapping)
        df_pred_combined["Actual Class"] = df_pred_combined["Actual Class"].map(class_mapping)

        col1, col2 = st.columns(2,vertical_alignment='top')
        
        count_test = y_test.value_counts()
        jumlah_paru = count_test[0] if 0 in count_test else 0
        jumlah_ekstra_paru = count_test[1] if 1 in count_test else 0
        
        with col1 :
            st.dataframe(df_pred_combined)
        
        with col2 :
            st.write("### Persebaran Data Testing Terhadap Target : ")
            st.write(f"Jumlah paru di data testing: {jumlah_paru}")
            st.write(f"Jumlah ekstra paru di data testing: {jumlah_ekstra_paru}")
        # Display the combined DataFrame with string labels
        
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred_perceptron
        

def show_evaluasi():
    st.title("Evaluasi Metode Perceptron")
    if 'y_test' in st.session_state and 'y_pred' in st.session_state : 
        y_test = st.session_state['y_test']
        y_pred_perceptron = st.session_state['y_pred']
        unique_classes = y_test.unique()
        c_matrix = confusion_matrix(y_test, y_pred_perceptron, labels=unique_classes)

        # Display confusion matrix in Streamlit
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=unique_classes).plot(ax=ax)
        plt.title("Confusion Matrix for Perceptron")
        st.pyplot(fig)
        
        st.markdown("---")
        
        
        st.write("### Rumus Untuk menentukan Akurasi, Recall, Presisi, dan F1 Score")
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.latex(r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}')
            
        
        with col2 :
            st.latex(r'Precision = \frac{TP}{TP + FP}')
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.latex(r'F1 Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}')
            
        with col2 :
            st.latex(r'Recall = \frac{TP}{TP + FN}')
            
        st.markdown("---")
        
        accuracy = accuracy_score(y_test, y_pred_perceptron) * 100
        precision = precision_score(y_test, y_pred_perceptron, average='weighted') * 100
        recall = recall_score(y_test, y_pred_perceptron, average='weighted') * 100
        f1 = f1_score(y_test, y_pred_perceptron, average='weighted') * 100
        
        
        st.write("### Performance Metrics for Perceptron Model")
        col1, col2 = st.columns(2,vertical_alignment='top')

        with col1 :
            st.write(f"#### Accuracy: {accuracy:.2f}%")
            
        with col2 :
            st.write(f"#### Precision: {precision:.2f}%")
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.write(f"#### Recall: {recall:.2f}%")
        
        with col2 :
            st.write(f"#### F1 Score: {f1:.2f}%")
        
        st.markdown("---")
        st.write("### Calssification Report")
        
        report = classification_report(y_test, y_pred_perceptron)
        
        st.markdown(f"```\n{report}\n```")

def show_testing():
    with open("model/perceptron_pickle", "rb") as r:
        perp = pickle.load(r)
        
    with open("model/minmax_scaler_pickle", "rb") as r:
        scaler = pickle.load(r)

    LABEL = ["Paru", "Ekstra Paru"]
    
    
    gender_display = {0: 'Perempuan', 1: 'Laki-laki'}
    ft_display = {0: 'Negatif', 1: 'Positif'}
    hiv_display = {0: 'Negatif', 1: 'Positif'}
    diabet_display = {0: 'tidak', 1: 'Ya'}
    tcm_display = {0: 'Rif Sensitif', 1: 'Negatif', 2: 'Rif resisten'}

    st.title('Prediction Form')

    age = st.number_input('Masukkan Umur', min_value=0, max_value=150, value=30)
    gender = st.selectbox('Pilih gender', ['Perempuan', 'Laki-laki'])
    ft = st.selectbox('Foto Toraks', ['Negatif', 'Positif'])
    hiv = st.selectbox('Status HIV', ['Negatif', 'Positif'])
    diabet = st.selectbox('Riwayat Diabetes', ['tidak', 'Ya'])
    tcm = st.selectbox('Hasil TCM', ['Rif Sensitif', 'Negatif', 'Rif resisten'])

    if st.button('Predict'):
        # Convert selectbox values to integer based on specified mappings
        gender_map = {'Perempuan': 0, 'Laki-laki': 1}
        ft_map = {'Negatif': 0, 'Positif': 1}
        hiv_map = {'Negatif': 0, 'Positif': 1}
        diabet_map = {'tidak': 0, 'Ya': 1}
        tcm_map = {'Rif Sensitif': 0, 'Negatif': 1, 'Rif resisten': 2}

        gender = gender_map[gender]
        ft = ft_map[ft]
        hiv = hiv_map[hiv]
        diabet = diabet_map[diabet]
        tcm = tcm_map[tcm]

        # Normalize age using loaded scaler
        features = np.array([[age, gender, ft, hiv, diabet, tcm]], dtype=np.float64)
        normalized_features = scaler.transform(features)

        newdata = normalized_features.tolist()
        result = perp.predict(newdata)
        result = LABEL[result[0]]

        col1, col2 = st.columns([2, 1], vertical_alignment='top')

        with col1:
            st.subheader('Input:')
            st.write(f"Umur : {age}")
            st.write(f"Jenis Kelamin : {gender_display[gender]}")
            st.write(f"Foto Toraks : {ft_display[ft]}")
            st.write(f"Status HIV : {hiv_display[hiv]}")
            st.write(f"Riwayat Diabetes : {diabet_display[diabet]}")
            st.write(f"Hasil TCM : {tcm_display[tcm]}")

        with col2:
            st.subheader('Prediction:')
            st.write(f":blue[{result}]")
            
        # st.write(f"Umur: {age}, Jenis Kelamin: {gender}, Foto Toraks: {ft}, Status HIV: {hiv}, Riwayat Diabetes: {diabet}, Hasil TCM: {tcm}")
    

if __name__ == "__main__":
    st.set_page_config(page_title="Perceptron", page_icon="images/perp.png")
    main()
