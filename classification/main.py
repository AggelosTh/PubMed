from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import eda_utils
from load_data import df
from mlmodel import SBertModel, sbert, device
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)

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
    
    # Get the word cloud buffer
    img_buffer = eda_utils.create_wordcloud(df)

    # Return the image as a response
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/label-count")
async def get_label_count():
    img_buffer = eda_utils.plot_label_count(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/label-correlation")
async def get_label_correlation():
    img_buffer = eda_utils.correlation_between_labels(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/most-common-mesh")
async def get_most_common_mesh():
    img_buffer = eda_utils.count_most_common_mesh(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/text-length-distribution")
async def get_text_length_distribution():
    img_buffer = eda_utils.draw_text_length_distribution(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.post("/predict")
async def predict(query: str):
    inputs = sbert.encode(query, convert_to_tensor=True, device=device)
    output = ml_model["answer"](inputs)
    predictions = (torch.sigmoid(output) > 0.5).float()
    list_of_predictions = predictions.cpu().numpy().tolist()
    labels = [label for label, prediction in zip(label_mapping, list_of_predictions) if prediction == 1]
    return {"Labels": labels}
