import uvicorn
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Path, UploadFile, File
import tensorflow as tf
import numpy as np
import aiofiles
import os
from io import BytesIO
from tensorflow.keras.preprocessing import image
from PIL import Image
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()  # create a new FastAPI app instance

# port = int(os.getenv("PORT"))
port = 8080

model = tf.keras.models.load_model('trial_model_1.h5')

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def somethingidk(file):
    print("classifier initiated")
    print(file)
    # predicting images
    img = image.load_img(file, target_size=(200, 200))


    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x]) 
    classes = model.predict(images) 
    print(classes[0]) 

    if classes[0] > 0.9:
        return 'normal'    
    
    return 'defect'

def somethingidk2(file):
    print("classifier initiated")
    print(type(file))
    # predicting images

    # path = file
    path = "_9973017b-e677-4377-b647-0bc2d6c899bd.jpg"
    img = image.load_img(path, target_size=(200,200))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    print(x)

    # images = np.vstack([x]) 
    # classes = model.predict(images) 
    # print(classes[0]) 


    # if classes[0] > 0.9:
    #     return 'normal'
    
    return 'defect'


@app.get("/")
def hello_world():
    return ("hello world")

# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}

@app.post("/")
async def post_endpoint(in_file: UploadFile=File(...)):
    out_file_path = "/"
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write

    return {"Result": "OK"}

@app.post("/2/")
async def image(image: UploadFile = File(...)):
    with open("destination.png", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    print(buffer)
    return {"filename": image.filename}

@app.post("/3/")
def image(image: UploadFile = File(...)):
    print(image.filename)
    print(type(image.filename))
    savefile = image.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    result = somethingidk2(savefile)
    return {result}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    result = somethingidk(image)
    return {result}

@app.post("/uploadfile2/")
async def create_upload_file(file: UploadFile):
    print(file)
    result = somethingidk2(file)
    return result

@app.post("/files/")
async def UploadImage(file: UploadFile):
    with open('image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'got it'
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)