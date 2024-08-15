import streamlit as st
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from wordcloud import WordCloud

st.title('Analisis Sentimen')
st.markdown('Menggunakan Algoritma TF-IDF')
st.sidebar.title('Dataset')

uploaded_file = st.sidebar.file_uploader('Upload dataset', type=['csv'])
if uploaded_file is not None:
    # Baca dataset
    data = pd.read_csv(uploaded_file)
    data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    # Tampil dataset awal
    st.sidebar.write('Dataset awal')
    st.sidebar.write(data)
    st.sidebar.write("Jumlah data : ", len(data))

    exp0 = st.expander("Tahap Preprocessing")
    tabs1, tabs2, tabs3, tabs4, tabs5, tabs6 = exp0.tabs(['Case Folding', 'Tokenizing', 'Penghapusan Tanda Baca', 'Stemming', 'Penggabungan Kata Sinonim dan kata baku', 'Penghapusan Stopwords'])

    # Case folding
    data['casefolded'] = data['text'].str.lower()
    tabs1.write(data['casefolded'])
    tabs1.write("Jumlah data :")
    tabs1.write(len(data['casefolded']))

    # Tokenizing
    data['tokens'] = data['casefolded'].apply(word_tokenize)
    tabs2.write(data['tokens'])
    tabs2.write("Jumlah data :")
    tabs2.write(len(data['tokens']))

    # Menghapus tanda baca
    def remove_punc(tokens):
        custom_punctuation = ['.', '..', '...', '....', '.....', '......', '.......']
        tokens_filtered = [token for token in tokens if token not in string.punctuation and token not in custom_punctuation]
        return tokens_filtered

    data['punc_removed'] = data['tokens'].apply(remove_punc)

    tabs3.write(data['punc_removed'])
    tabs3.write("Jumlah data :")
    tabs3.write(len(data['punc_removed']))

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming(tokens):
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    data['stemmed'] = data['punc_removed'].apply(stemming)
    tabs4.write(data['stemmed'])
    tabs4.write("Jumlah data :")
    tabs4.write(len(data['stemmed']))
    
    # Penggabungan kata
    def join_tokens(tokens):
        return ' '.join(tokens)
    data['clean_text'] = data['stemmed'].apply(join_tokens)


    # Buat kamus sinonim
    dict = {
        'tidak': ['ora', 'g', 'ngk', 'gx', 'gk', 'gak', 'ga', 'ngga', 'tdak', 'kaga'],
        'sampai': ['sampe'],
        'tidak bisa': ['gabisa', 'gbs'],
        'sudah': ['udah', 'udh'],
        'nomor': ['nomer'],
        'ok': ['oks', 'okk', 'oke'],
        'mantap': ['jos', 'mantab', 'mantabb', 'mantp', 'mantapp', 'mantappp', 'mantapppp', 'mantappppp', 'mantaaap'],
        'mantap betul': ['mantul', 'mantuls'],
        'tetap': ['tetep'],
        'malah': ['mlh'],
        'bisa': ['isa'],
        'notifikasi': ['notif'],
        'bantu': ['kabantu'],
        'tentang': ['tntg'],
        'terima kasih': ['trmksih', 'trimakasih'],
        'gunakan': ['gunain'],
        'mudah': ['gampang'],
        'tunggu': ['nunggu'],
        'terima': ['rima'],
        'berkali kali': ['verkali-kali'],
        'tahu': ['tau'],
        'belum': ['blm'],
        'sangat': ['sangaat', 'snagat', 'sangt2'],
        'aplikasi': ['applikasi'],
        'simpel': ['simple'],
        'baik': ['good'],
        'bermanfaat': ['nermanfaat'],
        'aja': ['ajh'],
        'saja': ['doang'],
        'uang': ['cuan'],
        'mengirim': ['ngirim'],
        'tahi': ['taeekkk', 'tai'],
        'paksa': ['maksa'],
        'obati': ['obatin'],
        'kirim': ['send'],
        'mudah mudah an': ['mudah2an'],
        'juga': ['jg'],
        'padahal': ['pdhl', 'pdahal'],
        'datang': ['dateng'],
        'membantu': ['membntu'],
        'kenapa': ['knp', 'kenap'],
        'masuk masuk': ['masuk2'],
        'bisa bisa': ['bisa2'],
        'antri': ['antre'],
        'eror': ['error', 'elor'],
        'daftar': ['daftr', 'dafta'],
        'kok': ['ko'],
        'tersangkut': ['stuck'],
        'susahnya': ['susahnyaaaa'],
        'otak': ['ngotak'],
        'entah': ['ntah'],
        'menghubungi': ['ngehubungi'],
        'kalau': ['klau'],
        'bagus': ['bgus'],
        'mah': ['ma'],
        'semoga': ['smoga'],
        'sip': ['sipp'],
        'verifikasi': ['perivikasi', 'verifikas'],
        'pakai': ['pke'],
        'di': ['d'],
        'makin': ['mangkin'],
        'dong': ['donk'],
        'perbaiki': ['perbaikin'],
        'tiba tiba': ['tiba2'],
        'lagi': ['lgi'],
        'gimana': ['gimna', 'gmn'],
        'bodoh': ['bloon'],
        'menyebalkan': ['nyebelin'],
        'aku': ['ak'],
        'unduh': ['download', 'donlwod'],
        'ubah': ['rubah'],
        'rekomendasi': ['recomen'],
        'coba': ['nyoba'],
        'sistem': ['sistim'],
        'fungsi': ['pungsi'],
        'setiap': ['tiap'],
        'bermanfaat': ['bemafaat'],
        'benar': ['bener'],
        'boleh': ['bole'],
        'anjing': ['ajg'],
        'setelah': ['stelah']
    }

    # Mengganti kata-kata dengan sinonimnya dan kata baku
    def replace_synonyms(text):
        words = text.split()
        replaced_words = []
        for word in words:
            for key, synonyms in dict.items():
                if word in synonyms:
                    replaced_words.append(key)
                    break
            else:
                replaced_words.append(word)
        return ' '.join(replaced_words)

    data['synonym'] = data['clean_text'].apply(replace_synonyms)


    tabs5.write(data['synonym'])
    tabs5.write("Jumlah data :")
    tabs5.write(len(data['synonym']))

    # Filtering
    sw = StopWordRemoverFactory()

    # Menambahkan stopword tambahan
    more_stopwords = ['dan', 'sekali', 'selalu', 'tpi', 'yg', 'tp', 'pas',
                      'banget', 'sangat', 'trs', 'aja', 'dah', 'nya', 'sih',
                      'di', 'jadi', 'sak', 'kan', 'nih', 'woy', 'dong', 'la',
                      'in', 'oy', 'nya', 'wah', 'lah', 'tuh', 'loll', 'bngt',
                      'bet', 'bangettt', 'nge', 'hei', 'ni']

    stopwords = sw.get_stop_words()+more_stopwords

    data['sw_removed'] = data['synonym'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))


    tabs6.write(data['sw_removed'])
    tabs6.write("Jumlah data :")
    tabs6.write(len(data['sw_removed']))

    data['processed'] = data['sw_removed'].sort_index(ascending=False)

    # Hapus data kosong
    data.dropna(subset=['processed'], inplace=True)

    # Tampil dataset setelah text preprocessing
    exp1 = st.expander("Hasil akhir dataset setelah text preprocessing")
    exp1.write(data['processed'])
    exp1.write("Jumlah data : ")
    exp1.write(len(data['processed']))

    # Mapping label menjadi kategori: positif, netral, dan negatif
    def map_label(label):
        if label == 0:
            return 'positif'
        else:
            return 'negatif'

    data['label'] = data['label'].apply(map_label)

    # Pembagian data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(data['processed'], data['label'], test_size=0.25, random_state=0)

    # Pembobotan TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Pelatihan model
    clf = SVC(kernel='linear')
    clf.fit(X_train_tfidf, y_train)

    # Klasifikasi menggunakan model
    y_pred = clf.predict(X_test_tfidf)

    st.header("Hasil Klasifikasi")

    # Evaluasi akurasi model
    accuracy = clf.score(X_test_tfidf, y_test)
    st.write("Akurasi model :", accuracy)
    
    # Evaluasi nilai precision
    precision_scores = precision_score(y_test, y_pred, labels=['positif', 'negatif'], average=None)

    # Evaluasi nilai recall
    recall_scores = recall_score(y_test, y_pred, labels=['positif', 'negatif'], average=None)

    # Kalkulasi F1 score
    f1_scores = f1_score(y_test, y_pred, labels=['positif', 'negatif'], average=None)

    # Menampilkan nilai precision dan recall
    st.write("Precision Score for positif : ", precision_scores[1])
    st.write("Precision Score for negatif : ", precision_scores[0])
    st.write("Recall Score for positif : ", recall_scores[1])
    st.write("Recall Score for negatif : ", recall_scores[0])

    # Menampilkan F1 score
    st.write("F1 Score for positif : ", f1_scores[0])
    st.write("F1 Score for negatif : ", f1_scores[1])

    exp2 = st.expander("Figure")
    tab1, tab2, tab3 = exp2.tabs(["F1 Score", "Confusion Matrix", "Wordcloud"])

    # Bar chart F1 score
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.bar(['positif', 'negatif'], f1_scores, color=['blue', 'orange'])
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    for i, score in enumerate(f1_scores):
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')
    ax=ax
    col1, col2 = tab1.columns([3,1])
    col1.pyplot(fig)
    col2.write("F1 Score for positif :")
    col2.write(f1_scores[0])
    col2.write("F1 Score for negatif :")
    col2.write(f1_scores[1])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Manampilkan heatmap confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positif', 'negatif'], yticklabels=['positif', 'negatif'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    ax=ax
    coll1, coll2 = tab2.columns([3,1])
    coll1.pyplot(fig)
    coll2.write("True Positive : ")
    coll2.write(cm[1,1])
    coll2.write("True Negative : ")
    coll2.write(cm[0,0])
    coll2.write("False Positive : ")
    coll2.write(cm[0,1])
    coll2.write("False Negative : ")
    coll2.write(cm[1,0])

    # Membuat wordcloud
    cleaned_text = ' '.join(data['processed'])
    wordcloud = WordCloud(width=600, height=450, background_color ='black').generate(cleaned_text)

    # Menampilkan wordcloud
    fig,ax = plt.subplots(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    ax=ax
    tab3.pyplot(fig)

    # Tampilkan teks berserta label prediksi dan label aktual
    exp3 = st.expander("Teks dan label hasil prediksi")
    df_result = pd.DataFrame({'Text': X_test, 'Actual Label': y_test, 'Predicted Label': y_pred})
    exp3.write(df_result)
    exp3.write("Jumlah data : ")
    exp3.write(len(df_result))

    # Tambahkan input teks dan prediksi sentimen
    st.header("Prediksi Sentimen dari Input Teks")
    input_text = st.text_area("Masukkan teks untuk diprediksi:")

    if st.button("Prediksi Sentimen"):
        if input_text:
            # Text preprocessing
            casefolded_text = input_text.lower()
            tokens = word_tokenize(casefolded_text)
            punc_removed = remove_punc(tokens)
            stemmed = stemming(punc_removed)
            joined_text = join_tokens(stemmed)
            synonym_replaced = replace_synonyms(joined_text)
            sw_removed = ' '.join([word for word in synonym_replaced.split() if word not in stopwords])

            # Pembobotan TF-IDF
            input_tfidf = vectorizer.transform([sw_removed])

            # Prediksi sentimen
            prediction = clf.predict(input_tfidf)
            st.write("Prediksi sentimen : ", prediction[0])
        else:
            st.write("Masukkan teks untuk diprediksi.")
