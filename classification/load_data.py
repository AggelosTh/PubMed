import pandas as pd
import numpy as np
import logging
import ast

logger = logging.getLogger(__name__)

df = pd.read_csv("PubMed_Multi_Label_Text_Classification.csv")
df = df.fillna(value=np.nan)
df = df.dropna()

columns = list(df.columns)
mesh_categories = columns[6:]

df['labels'] = list(df[mesh_categories].values)
df['Title'] = df['Title'].astype('string')
df['abstractText'] = df['abstractText'].astype('string')
df['text'] = df['Title'] + ' ' + df['abstractText']

print(df.head())
logger.info(df.info())

df['meshMajor'] = df['meshMajor'].apply(ast.literal_eval)

df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print(df['word_count'].describe())
