from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import eda_utils
from load_data import df
from mlmodel import SBertModel, sbert, device
import torch
from contextlib import asynccontextmanager

ml_model = {}

label_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model = SBertModel(384, 14)
    model.to(device=device)
    model.load_state_dict(torch.load('best_model_multilayers.pth'))
    model.eval()
    ml_model["answer"] = model
    yield
    # Clean up the ML models and release the resources
    ml_model.clear()

app = FastAPI(lifespan=lifespan, debug=True)

@app.get("/wordcloud")
async def get_wordcloud():
    """Craete wordcloud endpoint

    Returns:
        StreamingResponse: a streaming response containing a wordcloud image
    """
    img_buffer = eda_utils.create_wordcloud(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/label-count")
async def get_label_count():
    """Count the labels endpoint

    Returns:
        StreamingResponse: a streaming response containing a count of labels image
    """
    img_buffer = eda_utils.plot_label_count(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/label-correlation")
async def get_label_correlation():
    """Label correlation endpoint

    Returns:
        StreamingResponse: a streaming response containing a label correlation image
    """
    img_buffer = eda_utils.correlation_between_labels(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/most-common-mesh")
async def get_most_common_mesh():
    """Most common mesh endpoint

    Returns:
        StreamingResponse: a streaming response containing an image of the most common mesh
    """
    img_buffer = eda_utils.count_most_common_mesh(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/text-length-distribution")
async def get_text_length_distribution():
    """Text length distribution endpoint

    Returns:
        Streamingresponse: a streaming response containing an image of text length distribution
    """
    img_buffer = eda_utils.draw_text_length_distribution(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.post("/predict")
async def predict(query: str):
    """Inference endpoint which predicts the labels an article belongs

    Args:
        query (str): the input article/query

    Returns:
        dict: the predicted labels the article belongs
    """
    inputs = sbert.encode(query, convert_to_tensor=True, device=device)
    output = ml_model["answer"](inputs)
    predictions = (torch.sigmoid(output) > 0.5).float()
    list_of_predictions = predictions.cpu().numpy().tolist()
    labels = [label for label, prediction in zip(label_mapping, list_of_predictions) if prediction == 1]
    return {"Labels": labels}
