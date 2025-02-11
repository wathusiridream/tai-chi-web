from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from model import load_custom_model, process_video_for_web , calculate_matching_accuracy_dtw
import time

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'mov'}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดล
model = load_custom_model("model_segment9.keras")

##@title
# batch size (N)
# num_step (T)
# embedding size (d)
BATCH_SIZE =  2#@param {type:"integer"}
NUM_STEPS = 58 #@param {type:"integer"}
NUM_CONTEXT_STEPS =  2#@param {type:"integer"}
CONTEXT_STRIDE =  1#@param {type:"integer"}



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    # Process new video
    test_embedding = process_video_for_web(
        filepath, 
        model,
        num_context_steps=NUM_CONTEXT_STEPS,
        context_stride=CONTEXT_STRIDE,
        frames_per_batch=58
    )
    
    if test_embedding is None:
        return jsonify({"error": "Failed to process video"}), 400
    
    # Load reference embedding
    reference_path = os.path.join(os.getcwd(), "embeddings", "reference.npy")
    if not os.path.exists(reference_path):
        return jsonify({"error": "Reference embedding not found"}), 500
    
    reference_embedding = np.load(reference_path)
    
    # Calculate matching accuracy
    results = calculate_matching_accuracy_dtw(reference_embedding, test_embedding)

    # Add debug information
    results["debug_info"] = {
        "test_embedding_shape": test_embedding.shape,
        "reference_embedding_shape": reference_embedding.shape,
    }
    
    print(f"✅ API Response: {results}")  # ✅ ตรวจสอบค่าที่ Flask คืนกลับ

    return jsonify(results)
if __name__ == "__main__":
    app.run(debug=True)
