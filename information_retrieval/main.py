from fastapi import FastAPI
import elasticsearch_utils
app = FastAPI()


index_name = 'angelos_index'

@app.post('/create_index')
async def create_index(index_name:str) -> dict:
    """Creating index endpoint

    Args:
        index_name (str): the name of the index

    Returns:
        dict: the response message
    """
    response = elasticsearch_utils.create_index(index_name=index_name)
    return {"response": response}

@app.post('/delete_index')
async def delete_index(index_name: str) -> dict:
    """Deleting index endpoint

    Args:
        index_name (str): the name of the index

    Returns:
        dict: the rsponse message
    """
    response = elasticsearch_utils.delete_index(index_name=index_name)
    return {"response": response}


@app.post('/index_documents')
async def index_documents(index_name: str):
    """Index documents endpoint

    Args:
        index_name (str): the name of the index
    """
    elasticsearch_utils.index_documents(index_name=index_name)


@app.post('/retrieve_documents')
async def retrieve_documents(query: str, index_name: str, num_of_documents: int, rerank: bool = True)-> dict:
    """Retrieve documents endpoint

    Args:
        query (str): the provided query
        index_name (str): the name of the index
        num_of_documents (int): the number of the maximum documents to be retrieved
        rerank (bool, optional): apply the reranker. Defaults to True.

    Returns:
        dict: the documents retrieved
    """
    documents = elasticsearch_utils.retrieve_documents(
        query=query,
        index_name=index_name,
        num_of_documents=num_of_documents,
        rerank=rerank
    )
    return {"documents": documents}