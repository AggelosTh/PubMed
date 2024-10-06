import pandas as pd
from load_data import df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plot_label_count(df: pd.DataFrame):
    label_counts = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']].sum()

    label_df = pd.DataFrame(label_counts, columns=['Count']).reset_index()
    label_df.columns = ['Label', 'Count']

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Label', y='Count', data=label_df)
    plt.title('Count of Each Label in the Dataset')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def  count_most_common_mesh(df: pd.DataFrame):
    
    total_mesh = sum(df['meshMajor'], [])
    mesh_counts = Counter(total_mesh)
    most_common_mesh = mesh_counts.most_common(10)

    mesh, counts = zip(*most_common_mesh)
    plt.bar(mesh, counts)
    plt.xticks(rotation=45)
    plt.title('Top 10 Most Common Elements')
    plt.show()

count_most_common_mesh(df=df)    
