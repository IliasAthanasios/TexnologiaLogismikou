import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Συνάρτηση για την οπτικοποίηση μείωσης διάστασης
def visualize_2d(data, method):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    
    if labels.dtype == 'object':
        le = LabelEncoder()
        numeric_labels = le.fit_transform(labels)
    else:
        numeric_labels = labels

    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        # Ορισμός perplexity, πρέπει να είναι < αριθμού δειγμάτων
        perplexity = min(30, len(data) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity)
    
    try:
        reduced_data = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=numeric_labels, cmap='viridis')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{method} Visualization')

        if labels.dtype == 'object':
            cbar = plt.colorbar(scatter)
            cbar.set_ticks(range(len(le.classes_)))
            cbar.set_ticklabels(le.classes_)
        else:
            plt.colorbar(scatter)
        
        st.pyplot(plt)
    except ValueError as e:
        st.error(f"Error during {method} visualization: {e}")

# Συνάρτηση για Exploratory Data Analysis
def eda(data):
    st.write("Summary Statistics")
    st.write(data.describe())
    
    st.write("Class Distribution")
    st.bar_chart(data.iloc[:, -1].value_counts())

# Συνάρτηση για αλγόριθμο κατηγοριοποίησης
def classification_tab(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    
    if labels.dtype == 'object':
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    st.subheader("Classification Algorithms")
    
    k = st.slider('Select k for k-NN', 1, 20, 5)
    knn = KNeighborsClassifier(n_neighbors=k)
    
    try:
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        st.write(f'k-NN Accuracy: {acc_knn}')
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_knn))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_knn))
    except ValueError as e:
        st.error(f"Error during k-NN classification: {e}")
    
    # Προσθήκη άλλων αλγορίθμων κατηγοριοποίησης εδώ

# Συνάρτηση για αλγόριθμο ομαδοποίησης
def clustering_tab(data):
    features = data.iloc[:, :-1]
    
    st.subheader("Clustering Algorithms")
    
    k = st.slider('Select k for k-Means', 1, 20, 5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    try:
        kmeans.fit(features)
        clusters = kmeans.predict(features)
        silhouette_avg = silhouette_score(features, clusters)
        st.write(f'k-Means Silhouette Score: {silhouette_avg}')
    except ValueError as e:
        st.error(f"Error during k-Means clustering: {e}")
    
    # Προσθήκη άλλων αλγορίθμων ομαδοποίησης εδώ

def info_tab():
    st.title("Application Information")
    st.write("This web-based application is developed for data mining and analysis using Streamlit.")
    st.write("## Team Members")
    st.write("- Ilias Athanasios: Task A,C")
    st.write("- Laskaris Athanasios: Task B,C")
    st.write("## Project Tasks")
    st.write("### Task A: Data Loading and Preprocessing")
    st.write("### Task B: 2D Visualization and EDA")
    st.write("### Task C: Machine Learning Models and Evaluation")

def main():
    st.title("Data Analysis and Machine Learning App")

    # Φόρτωση δεδομένων
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.write("Data Preview")
        st.write(data.head())
        
        # Εμφάνιση στατιστικών και διαγραμμάτων EDA
        eda(data)

        # Tabs για PCA και t-SNE
        st.header("2D Visualization")
        method = st.selectbox("Choose a dimensionality reduction method", ['PCA', 't-SNE'])
        visualize_2d(data, method)

        # Tabs για αλγόριθμους Μηχανικής Μάθησης
        st.header("Machine Learning")
        tab1, tab2, tab3 = st.tabs(["Classification", "Clustering", "Info"])
        
        with tab1:
            classification_tab(data)
        
        with tab2:
            clustering_tab(data)
        
        with tab3:
            info_tab()

if __name__ == "__main__":
    main()
