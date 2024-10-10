from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer, CrossEncoder

es = Elasticsearch('http://localhost:9200')

if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Connection failed")

sbert = SentenceTransformer('all-mpnet-base-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

path_documents = 'dblpv11_sample.json'

def create_index(index_name: str) -> str:
    """Creates an elasticsearch index with dynamically mapping and sentence embeddings

    Args:
        index_name (str): the name of the index

    Returns:
        str: a message indicating if the index was created or not
    """
    index_exists = es.indices.exists(index=index_name)
    if index_exists.body:
        return f"Index '{index_name}' already exists."
    else:    
        mappings = {
            "mappings": {
                    "dynamic": True,
                    "embedding": {"type": "dense_vector", "dims": 384}
                }
            }
        es.indices.create(index=index_name, body=mappings, ignore=400)
        return f"Index '{index_name}' created"

def delete_index(index_name: str)->str:
    """Deletes an index if exists

    Args:
        index_name (str): the name of the index

    Returns:
        str: a message indicating if the index was deleted or not
    """

    index_exists = es.indices.exists(index=index_name)
    if index_exists.body:
        es.indices.delete(index=index_name)
        return f"Deleted '{index_name}' index"
    else:
        return f"The '{index_name}' index does not exist"


def index_documents(index_name: str, path: str = path_documents):
    """Index the documents in an elasticsearch index

    Args:
        index_name (str): the name of the index
        path (str, optional):the path of the json data. Defaults to path_documents.
    """
    # First create the index
    create_index(index_name=index_name)

    documents = []
    # Read the json data
    with open(path, 'r') as file:
        for line in file:
            documents.append(json.loads(line))    

    for doc in documents:
        if doc['abstract'] == '<UNK>':
            continue
        else:
            # Create the document embeddings
            doc['embedding'] = sbert.encode(doc['title'] + ' ' + doc['abstract'])
            res = es.index(index=index_name, body=doc)
            if res['result'] == "created":
                print(f"Document is indexed:\n{doc['title']}\n")
            else:
                print(f"Document failed to index:\n{doc['title']}\n")
    print("Total documents indexed: ", len(documents))

def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """Apply a reranker on documents based on a query

    Args:
        query (str): the provided query
        documents (list[dict]): the documents to be reranked

    Returns:
        list[dict]: the reranked documents
    """
    document_pairs = [[query, doc["title"] + ' ' + doc['abstract']] for doc in documents]
    scores = reranker.predict(document_pairs)
    reranked_documents_and_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    return [document for _, document in reranked_documents_and_scores]


def retrieve_documents(query: str, index_name: str, num_of_documents: int, rerank: bool = True)-> list[dict]:
    """Retrieve documents from the elasticsearch index based on a query

    Args:
        query (str): the provided query
        index_name (str): the name of the index
        num_of_documents (int): the maximum number of documents to retrieve
        rerank (bool, optional): apply a reranker. Defaults to True.

    Returns:
        list[dict]: the retrieved documents
    """
    documents = []
    query_embedding = sbert.encode(query)
    # Use knn query on the document embeddings
    search_query = {
        "knn": {
            "field": "embedding", 
            "query_vector": query_embedding, 
            "k": num_of_documents,  
            "num_candidates": 100
    }
    }
    response = es.knn_search(index=index_name, body=search_query)
    for res in response["hits"]["hits"]:
         documents.append(res['_source'])

    if rerank:
        # Apply the reranker
        documents = rerank_documents(query=query, documents=documents)
    return documents     
