from flask import Flask, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


@app.route("/static/<path:path>")
@cross_origin()
def send_frame(path):
    return send_from_directory('static', path)
