from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk import sent_tokenize

def cluster_data(cleaned_data):
    # Tokenize the text into sentences
    sentences = sent_tokenize(' '.join(cleaned_data))

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Set the number of clusters dynamically based on sentence count
    num_clusters = int(len(sentences) / 10) + 1  # Adjust as needed for the clustering size

    if num_clusters < 2:
        num_clusters = 2  # Ensure at least 2 clusters

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_matrix)

    # Get the best single representative sentence from each cluster
    clustered_indices = kmeans.predict(tfidf_matrix)
    best_representatives = []
    seen_clusters = set()

    for i, cluster_idx in enumerate(clustered_indices):
        if cluster_idx not in seen_clusters:
            cluster_sentences = [sentences[j] for j in range(len(clustered_indices)) if clustered_indices[j] == cluster_idx]
            best_representatives.append(cluster_sentences[0])  # Choose the first sentence as the representative
            seen_clusters.add(cluster_idx)

    return best_representatives
