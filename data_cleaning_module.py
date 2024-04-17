import re
import unicodedata
from nltk import sent_tokenize
import string

def clean_text(text):
    if not isinstance(text, (str, bytes)):
        raise ValueError("Input should be a string or bytes-like object.")
    # Text cleaning steps
    cleaned_sentences = []
    sentences = sent_tokenize(text)
    

    for sentence in sentences:
        # Text cleaning operations
        sentence = sentence.strip()
        
        # Convert text to lowercase
        sentence = sentence.lower()

        # Remove URLs
        sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence)
        
        # Remove email addresses
        sentence = re.sub(r'\S+@\S+', '', sentence)
        
        # Remove non-breaking spaces
        sentence = sentence.replace('\xa0', ' ')
        
        # Remove extra spaces
        sentence = ' '.join(sentence.split())
        
        # Remove HTML, XML, and JSON tags
        sentence = re.sub(r'<[^>]*>', '', sentence)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # Emojis
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            u"\U0001F700-\U0001F77F"  # Alchemical Symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            "]+", flags=re.UNICODE)
        sentence = emoji_pattern.sub(r'', sentence)
        
        # Remove non-ASCII characters
        sentence = ''.join(char for char in sentence if ord(char) < 128)
        
        # Remove diacritics (accents on characters)
        sentence = ''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')

        cleaned_sentences.append(sentence)

    return ' '.join(cleaned_sentences)

def clean_data(data):
    if isinstance(data, str):
        cleaned_text = clean_text(data)
        return cleaned_text
    elif isinstance(data, list):
        cleaned_data = [clean_text(sentence) for sentence in data]
        return cleaned_data
    elif hasattr(data, 'read') and callable(data.read):  # Check if it's a file-like object
        try:
            cleaned_text = clean_text(data.read().decode("utf-8"))
            return cleaned_text
        except Exception as e:
            print("Error reading file:", e)
            return None
    else:
        raise ValueError("Invalid data type for cleaning: " + str(type(data)))


