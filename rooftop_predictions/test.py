from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image
import base64
import io 

app = Flask(__name__)

def pil_2_b64(img_pil):
    arr = io.BytesIO()
    img_pil.save(arr, format="PNG")
    return base64.b64encode(arr.getvalue())

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = Image.open(request.files["upload_image"].stream).convert("RGB")
        print(img.size)
        print(img)
        img_b64 = pil_2_b64(img)
        data = jsonify({"image_data": img_b64})
        return render_template("test.html", **data)
    return render_template("test.html")

app.run(host='0.0.0.0', port=81)
