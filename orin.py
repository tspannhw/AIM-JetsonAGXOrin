from pymilvus import MilvusClient
import numpy as np
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import uuid
from time import sleep
from math import isnan
import time
import sys
import datetime
import subprocess
import sys
import os
import traceback
import math
import base64
import json
from time import gmtime, strftime
import random, string
import base64
import socket
import glob
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import multiprocessing
import cv2
import time
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# -----------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------

DIMENSION = 512

DATABASE_NAME = "./OrinEdgeAI.db"

COLLECTION_NAME = "OrinEdgeAI"

PATH = "/home/jetson/unstructureddata/images/"

slack_token = os.environ["SLACK_BOT_TOKEN"]

BLIP_MODEL = "Salesforce/blip-image-captioning-large"

# -----------------------------------------------------------------------------------------------
# Slack
# -----------------------------------------------------------------------------------------------

client = WebClient(token=slack_token)

# -----------------------------------------------------------------------------------------------
# Milvus Feature Extractor
# -----------------------------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

extractor = FeatureExtractor("resnet34")

# -----------------------------------------------------------------------------------------------
# Milvus Collection
# -----------------------------------------------------------------------------------------------

milvus_client = MilvusClient(DATABASE_NAME)

# -----------------------------------------------------------------------------------------------
# Create Milvus collection which includes the filepath of the image, and image embedding
# -----------------------------------------------------------------------------------------------

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='caption', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='currenttime', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)

milvus_client.create_collection(COLLECTION_NAME, DIMENSION, schema=schema, metric_type="COSINE", auto_id=True)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "vector", metric_type="COSINE")

milvus_client.create_index(COLLECTION_NAME, index_params)

# -----------------------------------------------------------------------------------------------
# OpenCV From Webcam
# -----------------------------------------------------------------------------------------------

cam = cv2.VideoCapture(0)
result, image = cam.read()
strfilename = PATH + 'orin{0}.jpg'.format(uuid.uuid4())

if result:
    cv2.imwrite(strfilename, image)
else:
    print("No image")

# -----------------------------------------------------------------------------------------------
# Metadata Fields
# -----------------------------------------------------------------------------------------------

currenttimeofsave = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
hostname = os.uname()[1]

print(hostname)
print(currenttimeofsave)

# -----------------------------------------------------------------------------
# BLIP Image Processing
# -----------------------------------------------------------------------------

processor = BlipProcessor.from_pretrained(BLIP_MODEL)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)

# raw_image = Image.open(strfilename)

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = (processor.decode(out[0], skip_special_tokens=True))

print(strfilename)
print("Caption: " + caption)

# -----------------------------------------------------------------------------
# Milvus insert
try:
    imageembedding = extractor(strfilename)
    milvus_client.insert( COLLECTION_NAME, {"vector": imageembedding, "currenttime":currenttimeofsave,
            "filename": strfilename, "caption": str(caption)})
    milvus_client.close()
except Exception as e:
    print("An error:", e)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Slack
try:
    response = client.chat_postMessage(
        channel="C06NE1FU6SE",
        text=(f"NVIDIA Orin Upload {hostname} {currenttimeofsave} {caption}")
    )
except SlackApiError as e:
    # You will get a SlackApiError if "ok" is False
    assert e.response["error"]

try:
    response = client.files_upload_v2(
        channel="C06NE1FU6SE",
        file=strfilename,
        title="NVIDIA Orin Upload " + str(caption),
        initial_comment="Live Camera image from NVIDIA Orin of " + str(caption),
    )
except SlackApiError as e:
    assert e.response["error"]
# Slack
# -----------------------------------------------------------------------------
