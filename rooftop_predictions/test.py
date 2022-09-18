from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image
import base64
import io 
import json
import time
import numpy as np
import torch
import net_relevant_region

N_DF = 16
N_CNV = 2
DROPOUT = 0.8
IMG_SIZE = 256
MODELS = "models/"

device = "cpu"

net = net_relevant_region.Relevant_Region(n_df=N_DF, n_cnv=N_CNV, p_dropout=DROPOUT)
net.load_state_dict(torch.load(MODELS + "net.pt", map_location="cpu"))
net = net.to(device)
net.eval()

app = Flask(__name__)

orig_img = None

def pil_2_b64(img_pil):
    arr = io.BytesIO()
    img_pil.save(arr, format="PNG")
    # return base64.b64encode(arr.getvalue())
    # return base64.b64encode(arr.getvalue()).decode("ascii")
    return base64.encodebytes(arr.getvalue()).decode("ascii")

@app.route('/get_predictions', methods=["GET", "POST"])
def get_predictions():
    print(orig_img) 
    if orig_img is None:
        return "predictions"
    img_np = np.array(orig_img, dtype=np.float32) / 256
    img_np -= img_np.min()
    img_np /= 1e-7 + img_np.max()
    print(img_np.shape)
    t_img = torch.from_numpy(img_np).view(-1, 3, IMG_SIZE, IMG_SIZE)
    print(t_img.size())
    t0 = time.time()
    out = net(t_img)[0]
    t1 = time.time()
    print(f"dt (inference): {t1-t0:.4f}")
    print(out.size())
    out_img = out.detach().permute(1,2,0).numpy()
    print(out_img.shape)
    out_img -= out_img.min()
    out_img /= (1e-7 + out_img.max())
    out_img_pil = Image.fromarray(np.uint8(255*out_img))
    return jsonify({"orig_img": pil_2_b64(orig_img), "heatmap": pil_2_b64(out_img_pil)})

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