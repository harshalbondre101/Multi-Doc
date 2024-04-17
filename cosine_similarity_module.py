from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

def calculate_cosine_similarity(cleaned_data):
    # Tokenize the text into sentences
    sentences = sent_tokenize(' '.join(cleaned_data))

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_threshold = 0.7

    cleaned_indices = set()
    for i in range(len(similarity_matrix)):
        if i not in cleaned_indices:
            similar_indices = [j for j in range(i+1, len(similarity_matrix)) if similarity_matrix[i][j] > similarity_threshold]
            cleaned_indices.update(similar_indices)

    unique_cleaned_sentences = [sentences[i] for i in range(len(sentences)) if i not in cleaned_indices]

    return unique_cleaned_sentences
