import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')


def clean_text_data(text: str) -> str:
    """Removes puncuation from string and unwanted characters

    Args:
        text (str): the input text

    Returns:
        str: the cleaned text
    """
    tokens_without_sw = [word for word in text.split() if not word in stop_words]
    joined_text = " ".join(word for word in tokens_without_sw)
    final_text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", joined_text)
    return final_text  

