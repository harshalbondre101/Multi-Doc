from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def model_topics(cleaned_data):
    # Vectorize the text using CountVectorizer
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(cleaned_data)

    # Set the number of topics dynamically based on data size
    num_topics = int(len(cleaned_data) / 100) + 1  # Adjust as needed for topic modeling size

    if num_topics < 2:
        num_topics = 2  # Ensure at least 2 topics

    # Perform LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(count_matrix)

    # Extract the top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-5:][::-1]  # Select top words for each topic
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        topic_info = f"Topic {topic_idx + 1}: {', '.join(top_words)}"
        topics.append(topic_info)

    # Return the array of top words for each topic
    return topics
