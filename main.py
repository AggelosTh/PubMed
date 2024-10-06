from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import eda_utils
from load_data import df

app = FastAPI()

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


@app.get("/most-common-mesh")
async def get_most_common_mesh():
    img_buffer = eda_utils.count_most_common_mesh(df)
    return StreamingResponse(img_buffer, media_type="image/png")


@app.get("/text-length-distribution")
async def get_text_length_distribution():
    img_buffer = eda_utils.draw_text_length_distribution(df)
    return StreamingResponse(img_buffer, media_type="image/png")