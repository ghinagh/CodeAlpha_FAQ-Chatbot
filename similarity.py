import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_faqs, load_faqs, spacy_text

# Load and preprocess FAQs
faqs = load_faqs("data/faqs.csv")
faqs = preprocess_faqs(faqs)

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faqs['spacy_question'])  # ✅ corrected column name

def get_answer(user_question):
    """
    Given a user question, return the best matching FAQ answer.
    """
    user_question_processed = spacy_text(user_question)  # ✅ correct function name
    user_vector = vectorizer.transform([user_question_processed])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    best_index = similarities.argmax()
    return faqs.iloc[best_index]['answer']  # ✅ corrected column name

# Example usage
if __name__ == "__main__":
    print(get_answer("When do you open?"))
