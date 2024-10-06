import seaborn as sns
import pandas as pd
import io
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import numpy as np
from wordcloud import WordCloud

def create_wordcloud(df: pd.DataFrame):

    text = ' '.join(df['text'].dropna())
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

def plot_label_count(df: pd.DataFrame):
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


def  count_most_common_mesh(df: pd.DataFrame):
    
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

def draw_text_length_distribution(df:pd.DataFrame):

    # _, _ = plt.subplots()
    count, bins_count = np.histogram([len(text.split()) for text in df['text']], bins=100)
    
    pdf = count / sum(count)
    
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.plot([200, 200], [0, 1], label="200 tokens", color='green')
    plt.plot([0, 1000], [0.90, 0.90], label="90%", color='black')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer