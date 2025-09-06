import pandas as pd
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Define stopwords set once
nltk_stopwords = set(stopwords.words("english"))

def load_faqs(csv_path):
    """Load FAQs from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def clean_text(text):
    """Lowercase, remove punctuation, remove stopwords using NLTK."""
    if not isinstance(text, str):  # handle NaN or non-string values
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in nltk_stopwords]  # remove stopwords
    return " ".join(tokens)  # join back into string

def spacy_text(text):
    """Tokenize and lemmatize using SpaCy."""
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def preprocess_faqs(df):
    """Preprocess FAQ questions and add cleaned columns."""
    df["clean_question"] = df["question"].apply(clean_text)
    df["spacy_question"] = df["question"].apply(spacy_text)
    return df  # ✅ return the DataFrame

# Example usage
if __name__ == "__main__":
    faqs = load_faqs("data/faqs.csv")
    faqs = preprocess_faqs(faqs)
    print(faqs.head())