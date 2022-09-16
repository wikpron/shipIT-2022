import torch
# import models
import numpy as np
import time
from flask import (
    Blueprint, flash, render_template, g, request, url_for
)

# net = models.RooftopsPreditor()
# net.load_state_dict(torch.load("saved_model_last.pt", map_location=torch.device("cpu")))
# net.eval()

bp = Blueprint('ml', __name__, url_prefix='/ml')

@bp.route('/predict', methods=('GET', 'POST'))
def predict():
    return render_template('ml/predict.html')

@bp.route('/get_labels', methods=('GET', 'POST'))
def get_labels():
    # global net
    return {
        "found_devices": False,
        "condifence": .5
    }
