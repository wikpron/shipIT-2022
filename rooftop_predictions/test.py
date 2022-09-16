from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    # return 'Web App with Python Flask!'
    return render_template("test.html")

app.run(host='0.0.0.0', port=81)
