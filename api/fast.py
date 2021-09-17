# $DELETE_BEGIN
import pytz

import io

import pandas as pd
import joblib
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from starlette.responses import StreamingResponse

from PIL import Image
import numpy as np
from tensorflow import cast, float32, int8

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/predict_old")
async def create_upload_file(file: UploadFile = File(...)):

    # print("\nreceived file:")
    # print(type(file))
    # print(file)

    # image_path = "image_api.png"

    # # write file to disk
    # with open(image_path, "wb") as f:
    #     f.write(file)

    contents = await file.read()

    img = tf.io.decode_png(tf.constant(contents), channels=3)
    decoded_image = tf.image.resize(img, [256, 256])

    new_model = tf.keras.models.load_model('gen_pix2pix_400_model_save.h5')

    expanded = tf.expand_dims(decoded_image, axis=0)

    result = new_model(expanded, training=False)

    # print(new_model.summary())

    print(result)

    # convert response from numpy to python type
    unnormalized = (result + 1) * 127.5
    preds = unnormalized.numpy()
    pred = preds[0]

    # .reshape(256, 256, 3)

    img = Image.fromarray(pred, 'RGB')
    img.save('my.png')

    return dict(prediction=pred)


@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):

    # read the content of the image
    contents = await file.read()

    img = tf.io.decode_png(tf.constant(contents), channels=3)

    # resize image
    decoded_image = tf.image.resize(
        img, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    decoded_image = cast(decoded_image, float32)

    expanded = tf.expand_dims((decoded_image / 127.5) - 1, axis=0)

    # load the model
    new_model = tf.keras.models.load_model('generator.h5')

    # make a prediction
    result = new_model(expanded, training=False)

    # convert pred to 0-255 image pixels
    np_image = cast((result[0] + 1) / 2.0 * 256, int8).numpy()

    pil_image = Image.fromarray(np_image, 'RGB')

    io_bytes = io.BytesIO()
    pil_image.save(io_bytes, format='PNG')

    image_bytes = io_bytes.getvalue()

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


# $DELETE_END
