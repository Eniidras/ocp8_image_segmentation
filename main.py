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

@app.route("/api", methods=["POST"])
def api():
    # change that : img_name = request.args.get("img_name")
    img_name = None
    pmask = modele.treat_image(img_name)
    res = Image.fromarray(pmask.astype(np.uint8))

    data = io.BytesIO()
    res.save(data, "PNG")
    encoded_res_data = base64.b64encode(data.getvalue())
    return # autre chose render_template("rendu.html", img_data=encoded_img_data.decode('utf-8'))

@app.route("/application_web/rendu", methods=["GET"])
def render():
    img_name = request.args.get("img_name")
    tmask = modele.return_mask(img_name)
    pmask = modele.treat_image(img_name)    

    # get the image
    img = modele.return_img(img_name)

    img_data = io.BytesIO()
    img.save(img_data, "PNG")
    encoded_img_data = base64.b64encode(img_data.getvalue())

    # get the true mask
    colored_tmask = modele.color_mask(tmask)

    tmask_data = io.BytesIO()
    colored_tmask.save(tmask_data, "PNG")
    encoded_tmask_data = base64.b64encode(tmask_data.getvalue())

    # get the predicted mask
    colored_pmask = modele.color_mask(pmask)

    pmask_data = io.BytesIO()
    colored_pmask.save(pmask_data, "PNG")
    encoded_pmask_data = base64.b64encode(pmask_data.getvalue())

    # get the superposition of the masks and the image
    img_with_tmask = modele.merge(tmask, img_name)

    img_with_tmask_data = io.BytesIO()
    img_with_tmask.save(img_with_tmask_data, "PNG")
    encoded_img_with_tmask_data = base64.b64encode(img_with_tmask_data.getvalue())

    img_with_pmask = modele.merge(pmask, img_name)

    img_with_pmask_data = io.BytesIO()
    img_with_pmask.save(img_with_pmask_data, "PNG")
    encoded_img_with_pmask_data = base64.b64encode(img_with_pmask_data.getvalue())

    return render_template("rendu.html",
        img_data=encoded_img_data.decode('utf-8'),
        tmask_data=encoded_tmask_data.decode('utf-8'),
        pmask_data=encoded_pmask_data.decode('utf-8'),
        img_with_tmask_data=encoded_img_with_tmask_data.decode('utf-8'),
        img_with_pmask_data=encoded_img_with_pmask_data.decode('utf-8'))

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
