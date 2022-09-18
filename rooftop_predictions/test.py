from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image
import base64
import io 
import json

app = Flask(__name__)

orig_img = None

def pil_2_b64(img_pil):
    arr = io.BytesIO()
    img_pil.save(arr, format="PNG")
    # img_pil.save("test.png")
    # return base64.b64encode(arr.getvalue())
    # return base64.b64encode(arr.getvalue()).decode("ascii")
    return base64.encodebytes(arr.getvalue()).decode("ascii")

@app.route('/get_predictions', methods=["GET", "POST"])
def get_predictions():
    print(orig_img) 
    if orig_img is None:
        return "predictions"
    return jsonify({"orig_img": pil_2_b64(orig_img)})

@app.route('/', methods=["GET", "POST"])
def index():
    global orig_img
    if request.method == "POST":
        img = Image.open(request.files["upload_image"].stream).convert("RGB")
        orig_img = img
        print(img.size)
        print(img)
        img_b64 = pil_2_b64(img)
        print(img_b64[:100])
        img_data = {"image_data": img_b64}
        # return render_template("test.html", **img_data)
        return render_template("test.html")
    return render_template("test.html")

app.run(host='0.0.0.0', port=81)