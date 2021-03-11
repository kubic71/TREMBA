import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
import numpy as np 
from PIL import Image




class GvisionWrapper:

    # For safety, max number of requests is by default set to conservative number
    def __init__(self, max_requests=500):
        
        self.n_request = 0
        self.max_requests = max_requests


    
    def __call__(self, img, dummy=False):
        img = img.transpose(1, 2, 0)
        if self.n_request >= self.max_requests:
            raise Exception("Google Vision max requests exceeded")

        self.n_request += 1


        if dummy:
            # [('Vertebrate', 0.9213282465934753),
            # ('Fin', 0.8927381634712219),
            # ('Seafood', 0.8372482657432556),
            # ('Fish', 0.8349323868751526),
            # ('Marine biology', 0.8116984367370605),
            # ('Fish products', 0.7809531688690186),
            # ('Ray-finned fish', 0.7554047107696533),
            # ('Tail', 0.7416104078292847),
            # ('Bony-fish', 0.5843339562416077),
            # ('Oily fish', 0.5834601521492004)]

            descriptions = ["Cat", "Small cat", "Tiger cat", "Plant"]
            scores = [0.95, 0.92, 0.8, 0.3]
            return descriptions, scores

        # Instantiates a client
        client = vision.ImageAnnotatorClient()

        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(".temp_img.png")

        # Loads the image into memory
        with io.open('.temp_img.png', 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image, max_results=100)
        labels = response.label_annotations

        print("Gvision request:", self.n_request)

        descriptions = [label.description for label in labels]
        scores = [label.score for label in labels]


        return descriptions, scores




