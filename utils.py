import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import random
import seaborn as sns
import pandas as pd
from load_data import df
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
random.seed(42)


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


def create_wordcloud(df: pd.DataFrame):

    text = ' '.join(df['text'].dropna())
    wordcloud = WordCloud().generate(text)

    # Display the wordcloud image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def plot_label_count(df: pd.DataFrame):
    label_counts = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']].sum()

    label_df = pd.DataFrame(label_counts, columns=['Count']).reset_index()
    label_df.columns = ['Label', 'Count']

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Label', y='Count', data=label_df, palette='viridis')
    plt.title('Count of Each Label in the Dataset')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def  count_most_common_mesh(df: pd.DataFrame):
    
    total_mesh = chain.from_iterable(df['meshMajor'])
    mesh_counts = Counter(total_mesh)
    most_common_mesh = mesh_counts.most_common(20)
    mesh, counts = zip(*most_common_mesh)
    plt.figure(figsize=(16, 16))
    sns.barplot(x=mesh, y=counts, palette='pastel')
    plt.xticks(rotation=45)
    plt.title('Top 20 Mesh terms')
    plt.show()

