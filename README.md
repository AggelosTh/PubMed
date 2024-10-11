# Multilabel Classification and Similar Article recommendation Application

This repository contains two key applications:
1. **Multilabel Classification** on the PubMed dataset.
2. **Similar Article recommendation system** using Elasticsearch as the database.

Both applications are built with **FastAPI** and packaged using **Conda** for the environment management.

## Table of Contents
- [Project Overview](#project-overview)
- [Multilabel Classification](#multilabel-classification)
- [Similar Article recommendation](#similar-article-recommendation)
- [Setup](#setup)
- [How to Run](#how-to-run)
- 
## Project Overview
This project integrates two major components: a multilabel classification task on biomedical literature and an Similar Article recommendation system to search and rank documents. Both systems are exposed as APIs via FastAPI.

1. **Multilabel Classification**: 
   - The model performs multilabel classification on abstracts from the PubMed dataset.
   - SBERT embeddings are used to create document representations, followed by the training and evaluation process.

2. **Similar Article recommendation**: 
   - This system uses Elasticsearch for indexing documents and to retrieve similar articles.
   - It employs embedding similarity retrieval methods and applies a ranker to the relevant documents for better results.

## Multilabel Classification
The **PubMed dataset** is used for training a model that predicts multiple labels for each biomedical abstract. The model is fine-tuned to predict categories of research papers, such as disease areas, research types, etc.

- **Model**: SBERT-based multilabel binary classification model.
- **Dataset**: PubMed abstracts.
- **Goal**: Assign multiple categories (labels) to each abstract.
  
### Key Steps:
1. Preprocess the PubMed dataset (titles, abstracts).
2. Generate SBERT embeddings for each title and abstract.
3. Train a multilabel classifier.
4. Evaluate the trained model
5. Expose the model via FastAPI for predictions.

### API Endpoints for Multilabel Classification

- `GET /wordcloud`: Generates a word cloud image from the dataset.
- `GET /label-count`: Returns a label count image.
- `GET /label-correlation`: Shows the correlation between different labels in an image.
- `GET /most-common-mesh`: Displays the most common MeSH terms in an image.
- `GET /text-length-distribution`: Returns an image showing the distribution of text lengths in the dataset.
- `POST /predict`: Predicts the labels for a given query (abstract) and returns them as a list.

## Similar Article recommendation
The Similar Article recommendation system indexes the Aminer DBLPv11 dataset abstracts using **Elasticsearch**. It allows users to search for similar articles and retrieve relevant ones, based on a given title and abstract.

- **Database**: Elasticsearch.
- **Indexing**: Documents (titles and abstracts) are indexed in Elasticsearch.
- **Retrieval Methods**: Emebddings similarity with knn and reranker.
  
### Key Steps:
1. Index the dataset into Elasticsearch.
2. Implement retrieval methods (Embeddings similarity, reranker).
3. Expose search endpoints via FastAPI.

### API Endpoints for Similar Article recommendation

- `POST /create_index`: Creates a new index in Elasticsearch.
    - **Input**: `index_name` (string) – the name of the index to create.
    - **Output**: Response indicating whether the index was successfully created.

- `POST /delete_index`: Deletes an existing index in Elasticsearch.
    - **Input**: `index_name` (string) – the name of the index to delete.
    - **Output**: Response indicating whether the index was successfully deleted.

- `POST /index_documents`: Indexes the Aminer DBLPv11 articles (titles and abstracts) into the specified index.
    - **Input**: `index_name` (string) – the name of the index where documents will be stored.

- `POST /retrieve_documents`: Retrieves documents from Elasticsearch based on a given title and abstract.
    - **Input**:
        - `title` (string) – the title of the article.
        - `abstract` (string) – the abstract of the article.
        - `index_name` (string) – the name of the index to retrieve from.
        - `num_of_documents` (int) – the number of documents to retrieve.
        - `rerank` (bool, optional) – whether to rerank the results using a reranker (default is `True`).
    - **Output**: A list of documents matching the query.

## Setting up the Conda Environment

Follow the steps below to create the Conda environment for this project.

### 1. Create the Conda Environment

To create the environment from the provided `environment.yml` file:

1. Open **Anaconda Prompt** or **Command Prompt**.
2. Navigate to the directory where the `environment.yml` file is located.
3. Run the following command to create the environment:

   ```bash
   conda env create -f environment.yml
