from sklearn.feature_extraction.text import TfidfVectorizer

def identify_important_parts(cleaned_data):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_data)

    # Identify important parts based on TF-IDF scores
    max_tfidf_indices = tfidf_matrix.argmax(axis=0)  # Get the index with the maximum TF-IDF score

    important_parts = [cleaned_data[i] for i in max_tfidf_indices.A1]  # A1 to convert to 1D array

    return important_parts
