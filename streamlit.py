import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Τίτλος εφαρμογής
st.title('Data Mining and Analysis Application')

# Φόρτωση Δεδομένων
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    st.write("Loaded Data")
    st.write(data)
    
    if 'label' in data.columns:
        features = data.columns[:-1]
        labels = data.columns[-1]
    else:
        st.error("The uploaded file does not contain a 'label' column")

    # 2D Visualization Tab
    st.header("2D Visualization")
    visualization_option = st.selectbox("Select a visualization method", ["PCA", "t-SNE"])

    def pca_visualization(data):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data.iloc[:, :-1])
    
    # Αντιστοίχιση κατηγοριών σε αριθμητικές τιμές για το χρωματισμό
        labels = data.iloc[:, -1]
        unique_labels = labels.unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = labels.map(label_mapping)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=numeric_labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')

    # Προσθήκη του χρωματικού μπαρ για την αντιστοίχιση κατηγοριών
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([label_mapping[label] for label in unique_labels])
    cbar.set_ticklabels(unique_labels)

    st.pyplot(plt)

# Φόρτωση δεδομένων από το CSV
    data = pd.read_csv('sample_data.csv')  # Αντικαταστήστε το με την πραγματική φόρτωση δεδομένων
    st.write("Data Preview")
    st.write(data.head())

    st.write("Labels Preview")
    st.write(data.iloc[:, -1].unique())

    pca_visualization(data)

    # Classification Tab
    st.header("Classification")
    classification_option = st.selectbox("Select a classification algorithm", ["k-NN", "SVM"])

    def knn_classification(data):
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=42)
        k = st.sidebar.slider('k', 1, 15, 5)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'k-NN Accuracy: {accuracy:.2f}')

    def svm_classification(data):
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=42)
        c = st.sidebar.slider('C', 0.1, 10.0, 1.0)
        svm = SVC(C=c)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'SVM Accuracy: {accuracy:.2f}')

    if classification_option == "k-NN":
        knn_classification(data)
    elif classification_option == "SVM":
        svm_classification(data)

    # Clustering Tab
    st.header("Clustering")
    clustering_option = st.selectbox("Select a clustering algorithm", ["k-means", "DBSCAN"])

    def kmeans_clustering(data):
        k = st.sidebar.slider('k', 1, 10, 3)
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(data.iloc[:, :-1])
        st.write(f'Cluster centers: {kmeans.cluster_centers_}')
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters)
        st.pyplot(plt)
        score = silhouette_score(data.iloc[:, :-1], clusters)
        st.write(f'Silhouette Score: {score:.2f}')

    def dbscan_clustering(data):
        eps = st.sidebar.slider('eps', 0.1, 10.0, 0.5)
        dbscan = DBSCAN(eps=eps)
        clusters = dbscan.fit_predict(data.iloc[:, :-1])
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters)
        st.pyplot(plt)
        if len(set(clusters)) > 1:
            score = silhouette_score(data.iloc[:, :-1], clusters)
            st.write(f'Silhouette Score: {score:.2f}')
        else:
            st.write("Silhouette Score: Not applicable (only one cluster)")

    if clustering_option == "k-means":
        kmeans_clustering(data)
    elif clustering_option == "DBSCAN":
        dbscan_clustering(data)
