import seaborn as sns
import pandas as pd
import io
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import numpy as np
from wordcloud import WordCloud
from utils import clean_text_data


def create_wordcloud(df: pd.DataFrame) -> io.BytesIO:
    """Creates a wordcloud image from the texts of a dataframe

    Args:
        df (pd.DataFrame): the input dataframe

    Returns:
        io.BytesIO: BytesIO object containing the wordcloud image in PNG format.
    """
    # Remove stopwords and clean the text
    df['clean_text'] = df['text'].apply(clean_text_data)
    text = ' '.join(df['clean_text'].dropna())
    wordcloud = WordCloud().generate(text)

    buffer = io.BytesIO()

    # Display the wordcloud image:
    plt.figure(figsize=(16, 16))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer

def plot_label_count(df: pd.DataFrame) -> io.BytesIO:
    """Creates an image with the frequency of the target labels

    Args:
        df (pd.DataFrame): the input dataframe

    Returns:
        io.BytesIO: BytesIO object containing the label count image in PNG format.
    """
    label_counts = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']].sum()

    label_df = pd.DataFrame(label_counts, columns=['Count']).reset_index()
    label_df.columns = ['Label', 'Count']

    plt.figure(figsize=(16, 16))
    sns.barplot(x='Label', y='Count', data=label_df, palette='viridis')
    plt.title('Count of Each Label in the Dataset')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer


def correlation_between_labels(df: pd.DataFrame) -> io.BytesIO:
    """Creates an image of the correlation between the target labels

    Args:
        df (pd.DataFrame): the iput dataframe

    Returns:
        io.BytesIO: BytesIO object containing the label correlation image in PNG format.
    """
    label_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
    label_df = df[label_columns]

    # Compute correlation matrix
    correlation_matrix = label_df.corr()

    # Plot heatmap
    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Labels')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer

def  count_most_common_mesh(df: pd.DataFrame) -> io.BytesIO:
    """Creates an image containing the frequency of the most common mesh

    Args:
        df (pd.DataFrame): the input dataframe

    Returns:
        io.BytesIO: BytesIO object containing the frequency of the most common mesh image in PNG format.
    """
    
    total_mesh = chain.from_iterable(df['meshMajor'])
    mesh_counts = Counter(total_mesh)
    most_common_mesh = mesh_counts.most_common(20)
    mesh, counts = zip(*most_common_mesh)
    plt.figure(figsize=(16, 16))
    sns.barplot(x=mesh, y=counts, palette='pastel')
    plt.xticks(rotation=45)
    plt.title('Top 20 Mesh terms')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer

def draw_text_length_distribution(df:pd.DataFrame) -> io.BytesIO:
    """Creates an image with the text length distribution

    Args:
        df (pd.DataFrame): the imput dataframe

    Returns:
        io.BytesIO: BytesIO object containing the text length distribution image in PNG format.
    """

    count, bins_count = np.histogram([len(text.split()) for text in df['text']], bins=100)
    
    pdf = count / sum(count)
    
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.plot([300, 300], [0, 1], label="300 tokens", color='green')
    plt.plot([0, 1000], [0.90, 0.90], label="90%", color='black')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer
