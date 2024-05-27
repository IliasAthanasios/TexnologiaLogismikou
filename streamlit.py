import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


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


def eda(data):
    st.write("Summary Statistics")
    st.write(data.describe())
    
    st.write("Class Distribution")
    st.bar_chart(data.iloc[:, -1].value_counts())


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
    except ValueError as e:
        st.error(f"Error during k-NN classification: {e}")
    
    


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
    
    

def main():
    st.title("Data Analysis and Machine Learning App")

    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.write("Data Preview")
        st.write(data.head())
        
        
        eda(data)

        
        st.header("2D Visualization")
        method = st.selectbox("Choose a dimensionality reduction method", ['PCA', 't-SNE'])
        visualize_2d(data, method)

        
        st.header("Machine Learning")
        tab1, tab2 = st.tabs(["Classification", "Clustering"])
        
        with tab1:
            classification_tab(data)
        
        with tab2:
            clustering_tab(data)

if __name__ == "__main__":
    main()
