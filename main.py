import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()  # create a new FastAPI app instance

# port = int(os.getenv("PORT"))
port = 8080

model = tf.keras.models.load_model('trial_model_1.h5')

def somethingidk2(file):
    img = image.load_img(file, target_size=(200,200))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x]) 
    classes = model.predict(images) 
    print(classes[0]) 

    if classes[0] > 0.9:
        return 'normal'
    
    return 'defect'

@app.get("/")
def hello_world():
    return ("hello world")

@app.post("/3/")
def classifybean(input: UploadFile = File(...)):
    print(input.filename)
    print(type(input.filename))
    savefile = input.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)
    result = somethingidk2(savefile)
    os.remove(savefile)
    return {result}
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)