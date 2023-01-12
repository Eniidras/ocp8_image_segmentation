from flask import Flask, jsonify, render_template, request, flash, send_from_directory, make_response #, redirect
import modele
import os
import cv2

import numpy as np

import io
import base64

from PIL import Image

app = Flask(__name__, static_url_path='/static')
app.secret_key = "T0to_na_p4s_d0rm1"

@app.route("/", methods = ['GET'])
def home():
    return render_template('home.html')

@app.route("/application_web", methods=["GET"])
def application():
    imgs = os.listdir("static/images/")
    for img in imgs:
        flash(img, "img")
    return render_template('application_web.html')

@app.route("/application_web/rendu", methods=["GET"])
def render():
    img_name = request.args.get("img_name")
    mask = modele.treat_image(img_name)
    mask = modele.merge(mask, img_name)

    data = io.BytesIO()
    mask.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template("rendu.html", img_data=encoded_img_data.decode('utf-8'))

@app.route("/application_web/rendu_brut", methods=["GET"])
def render_raw():
    img_name = request.args.get("img_name")
    img = modele.treat_image(img_name)

    im_pil = Image.fromarray(img.astype(np.uint8))

    data = io.BytesIO()
    im_pil.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template("rendu.html", img_data=encoded_img_data.decode('utf-8'))


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
