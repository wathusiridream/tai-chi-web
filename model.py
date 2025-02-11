import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dense, Dropout, Lambda # type: ignore
import keras
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean , cosine
from werkzeug.utils import secure_filename

import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# เลือกเฉพาะจุดที่ต้องการ (MediaPipe ใช้ index เริ่มที่ 0)
SELECTED_LANDMARKS = [7, 8, 11, 12, 13, 14, 15, 16, 
                       17, 18, 19, 20, 21, 22, 23, 24, 
                       25, 26, 27, 28, 29, 30, 31, 32]

@keras.saving.register_keras_serializable(package="MyCustomModel")
class TCCLandmarkEmbedder(tf.keras.Model):
    def __init__(
        self,
        embedding_size,
        num_steps,
        num_context_steps,
        filters=64,
        kernel_size=3,
        num_layers=3,
        dropout_rate=0.1,
        name="TCCLandmarkEmbedder",
        trainable=True,
        **kwargs,
    ):
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.embedding_size = embedding_size
        self.num_context_steps = num_context_steps
        self.num_steps = num_steps
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_landmarks = 24

        # Landmark processing with TCN
        self.tcn_layers = [
            tf.keras.Sequential([
                Conv1D(filters, kernel_size, padding="causal", dilation_rate=2 ** i, activation="relu"),
                BatchNormalization(),
                Dropout(dropout_rate)
            ]) for i in range(num_layers)
        ]

        # Final embedding layers
        self.embedding_layers = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(embedding_size),
            Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_size": self.embedding_size,
            "num_steps": self.num_steps,
            "num_context_steps": self.num_context_steps,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, steps=None, training=False):
        # Reshape inputs
        batch_size = tf.shape(inputs)[0]
        input_timesteps = inputs.shape[1]
        num_landmarks = inputs.shape[2]
        num_features = tf.shape(inputs)[2]  # แทนการใช้ num_landmarks * 2
        #x = tf.reshape(inputs, [batch_size, input_timesteps, num_landmarks * 2])
        x = tf.reshape(inputs, [batch_size, input_timesteps, num_features])
        # Generate default steps if not provided
        if steps is None:
            steps = self.generate_default_steps(batch_size, input_timesteps, self.num_steps)

        # Apply TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x, training=training)

        # Gather features at steps
        steps_indices = tf.cast(steps, tf.int32)
        batch_indices = tf.expand_dims(tf.range(batch_size), 1)
        batch_indices = tf.tile(batch_indices, [1, self.num_steps])
        indices = tf.stack([batch_indices, steps_indices], axis=-1)
        landmarks_at_steps = tf.gather_nd(x, indices)

        # Generate final embeddings
        final_embeddings = self.embedding_layers(landmarks_at_steps, training=training)
        return final_embeddings

    def generate_default_steps(self, batch_size, input_timesteps, target_timesteps):
        step_size = input_timesteps // target_timesteps
        default_steps = tf.range(0, input_timesteps, step_size)[:target_timesteps]
        return tf.tile(tf.expand_dims(default_steps, 0), [batch_size, 1])

##@title
# batch size (N)
# num_step (T)
# embedding size (d)
BATCH_SIZE =  2#@param {type:"integer"}
NUM_STEPS = 58 #@param {type:"integer"}
NUM_CONTEXT_STEPS =  2#@param {type:"integer"}
CONTEXT_STRIDE =  1#@param {type:"integer"}

LOSS_TYPE = 'regression_mse_var' #@param ["regression_mse_var", "regression_mse", "regression_huber", "classification"]
# STOCHASTIC_MATCHING = True #@param ["False", "True"] {type:"raw"}
SIMILARITY_TYPE = 'l2' #@param ["l2", "cosine"]
EMBEDDING_SIZE =  128 #@param {type:"integer"}
TEMPERATURE = 0.1 #@param {type:"number"}
LABEL_SMOOTHING = 0.0 #@param {type:"slider", min:0, max:1, step:0.05}
VARIANCE_LAMBDA = 0.001 #@param {type:"number"}
HUBER_DELTA = 0.1 #@param {type:"number"}
NORMALIZE_INDICES = True #@param ["False", "True"] {type:"raw"}
NORMALIZE_EMBEDDINGS = False #@param ["False", "True"] {type:"raw"}

# CYCLE_LENGTH = 8 #@param {type:"integer"}
# NUM_CYCLES = 10 #@param {type:"integer"}

LEARNING_RATE = 1e-4 #@param {type:"number"}
DEBUG = True #@param ["False", "True"] {type:"raw"}

NUM_LANDMARKS = 24 #@param {type:"number"}

# โหลดโมเดล
def load_custom_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"TCCLandmarkEmbedder": TCCLandmarkEmbedder}
    )

# สกัด coordinates
def extract_landmarks(image):
    """ ดึงเฉพาะ landmark ที่ต้องการ """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        all_landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])  # (33,2)
        selected_landmarks = all_landmarks[SELECTED_LANDMARKS]  # เลือกเฉพาะจุดที่ต้องการ
        return selected_landmarks.flatten()  # (24,2) -> (48,)
    else:
        return np.zeros(len(SELECTED_LANDMARKS) * 2)  # ถ้าไม่มี landmark ให้ใส่ 0
    
# ประมวลผลวิดีโอ
def process_video_for_web(video_path, model, num_context_steps=2, context_stride=1, frames_per_batch=58):
    """
    Web-optimized version of video processing based on original testing code
    """
    # อ่านวิดีโอและดึง landmarks
    landmarks_list = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = extract_landmarks(frame)
        if not np.all(landmarks == 0):  # กรองเฟรมที่ไม่พบ landmarks
            landmarks_list.append(landmarks)
    
    cap.release()

    if len(landmarks_list) == 0:
        print("❌ No valid landmarks extracted!")
        return None

    # แปลงเป็น numpy array
    landmarks_array = np.array(landmarks_list)  # Shape (T, 48)
    seq_len = len(landmarks_array)
    
    # Normalize landmarks coordinates
    def normalize_sequence(sequence):
        x = sequence[:, ::2]  # คู่
        y = sequence[:, 1::2]  # คี่
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + 1e-6) - 1.0
        y_norm = 2.0 * (y - y_min) / (y_max - y_min + 1e-6) - 1.0
        
        normalized = np.zeros_like(sequence)
        normalized[:, ::2] = x_norm
        normalized[:, 1::2] = y_norm
        return normalized

    landmarks_array = normalize_sequence(landmarks_array)
    
    # Process in batches like the original code
    embeddings = []
    num_batches = int(np.ceil(float(seq_len) / frames_per_batch))
    
    for batch_idx in range(num_batches):
        # Define main steps for this batch
        steps = np.arange(batch_idx * frames_per_batch, (batch_idx + 1) * frames_per_batch)
        steps = np.clip(steps, 0, seq_len - 1)
        
        # Create context steps
        def get_context_steps(step):
            return tf.clip_by_value(
                tf.range(step - (num_context_steps - 1) * context_stride,
                         step + context_stride,
                         context_stride),
                0, seq_len - 1)
        
        # Generate context steps for all main steps
        steps_with_context = tf.reshape(
            tf.map_fn(get_context_steps, tf.convert_to_tensor(steps, dtype=tf.int32)),
            [-1]
        )
        
        # Gather frames with context
        frames = tf.gather(landmarks_array, steps_with_context)
        frames = tf.cast(frames, tf.float32)
        frames = tf.expand_dims(frames, 0)  # Add batch dimension
        
        # Get embeddings for this batch
        batch_embs = model(frames, training=False).numpy()[0]
        embeddings.extend(batch_embs)
    
    # Trim to match sequence length
    embeddings = embeddings[:seq_len]
    return np.array(embeddings)

# คำนวณ Similarity Score
def calculate_similarity(embedding1, embedding2):
    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(cosine_similarity * 100)

'''
def calculate_matching_accuracy_dtw(embs1, embs2, threshold):
    """
    คำนวณความแม่นยำในการจับคู่ระหว่าง embeddings ของ 2 วิดีโอ โดยใช้ DTW
    """
    if embs1.shape[1] != embs2.shape[1]:
        return {"error": "Embedding dimensions do not match"}

    start_time = time.time()

    # 🔹 คำนวณ DTW
    start_time_dtw = time.time()
    distance, path = fastdtw(embs1, embs2, dist=euclidean)
    end_time_dtw = time.time()
    print(f"DTW Calculation Time: {end_time_dtw - start_time_dtw:.4f} seconds")

    # 🔹 จัดรูปแบบ Path ของ DTW
    path = np.array(path).T
    matched_frames_dtw = path[1]

    # 🔹 คำนวณระยะ L2 สำหรับแต่ละคู่ที่ DTW จับคู่
    distances = []
    for i, j in enumerate(matched_frames_dtw):
        dist = np.linalg.norm(embs1[i] - embs2[j])
        distances.append(dist)

    # 🔹 คำนวณ Matching Accuracy
    matched_count = sum(1 for dist in distances if dist <= threshold)
    total_frames1 = len(embs1)

    overall_matching_accuracy = (matched_count / total_frames1) * 100

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.4f} seconds")

    return {
        "overall_matching_accuracy": overall_matching_accuracy,
        "total_frames_video1": total_frames1,
        "total_frames_video2": len(embs2),
        "matched_frames": matched_count,
        "dtw_distance": distance,
        "dtw_path": path.tolist(),
        "frame_matching_details": [
            {
                "frame": int(i),
                "matched_to": int(j),
                "distance": float(dist),
                "is_matched": bool(dist <= threshold)  # ✅ แปลงเป็น Python Boolean
            }
            for i, j, dist in zip(range(total_frames1), matched_frames_dtw, distances)
        ]
    }
'''
def dist_fn(x, y):
  print('function : dist_fn')
  dist = np.sum((x-y)**2)
  return dist

'''def calculate_matching_accuracy_dtw(embs1, embs2):
    """ คำนวณความแม่นยำในการจับคู่ระหว่าง embeddings ของ 2 วิดีโอ โดยใช้ DTW """
    

    start_time = time.time()

    # ✅ ใช้ fastdtw() คำนวณ DTW Distance
    dtw_distance, path = fastdtw(embs1, embs2, dist=euclidean)
    path = np.array(path).T  # แปลงเป็น (2, num_matches)

    # ✅ ใช้ np.unique() เพื่อเลือกเฟรมที่แมตช์กันครั้งแรก
    _, uix = np.unique(path[0], return_index=True)
    matched_frames_dtw = path[1][uix]

    # ✅ คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของระยะห่าง L2
    distances = [tf.norm(embs1[int(i)] - embs2[int(j)]).numpy() for i, j in zip(range(len(embs1)), matched_frames_dtw)]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # ✅ ใช้ค่าเฉลี่ย + Standard Deviation เป็น Threshold Dynamic
    threshold = mean_distance + std_distance  # 🔹 ปรับให้เข้มงวดขึ้น

    total_frames1 = int(len(embs1))
    total_frames2 = int(len(embs2))

    # ✅ คำนวณ Matching Accuracy รายเฟรม
    frame_matching_accuracy = {
        f'Frame {int(i)}': {
            'matched_to_frame': int(matched_idx),
            'distance': float(distance),
            'is_matched': bool(distance <= threshold)
        }
        for i, (matched_idx, distance) in enumerate(zip(matched_frames_dtw, distances))
    }

    # ✅ คำนวณจำนวนเฟรมที่ Match จริงๆ
    matched_count = sum(1 for result in frame_matching_accuracy.values() if result['is_matched'])

    # ✅ เปลี่ยนตัวหารให้เป็นจำนวนเฟรมที่จับคู่ได้จริง
    overall_matching_accuracy = (matched_count / min(len(embs1), len(embs2))) * 100

    end_time = time.time()
    print(f"✅ การทำงาน calculate_matching_accuracy_dtw ใช้เวลา: {end_time - start_time:.4f} วินาที")

    return {
        'frame_matching_details': frame_matching_accuracy,
        'overall_matching_accuracy': overall_matching_accuracy,
        'dtw_distance': float(dtw_distance),
        'total_frames_video1': total_frames1,
        'total_frames_video2': total_frames2,
        'matched_frames': matched_count,
        'dtw_path': path.tolist()
    }
'''

def calculate_matching_accuracy_dtw(embs1, embs2):
    """ 
    คำนวณความแม่นยำในการจับคู่ระหว่าง embeddings ของ 2 วิดีโอ โดยใช้ DTW 
    และการปรับปรุงประสิทธิภาพ
    """
    start_time = time.time()

    # Normalize embeddings
    embs1_normalized = tf.keras.utils.normalize(embs1)
    embs2_normalized = tf.keras.utils.normalize(embs2)

    # คำนวณ DTW Distance ด้วย cosine similarity
    dtw_distance, path = fastdtw(embs1_normalized, embs2_normalized, dist=cosine)
    path = np.array(path).T  # แปลงเป็น (2, num_matches)

    # เลือกเฟรมที่แมตช์กันครั้งแรก
    _, uix = np.unique(path[0], return_index=True)
    matched_frames_dtw = path[1][uix]

    # คำนวณระยะทางสำหรับเฟรมที่จับคู่กัน
    distances = [cosine(embs1_normalized[int(i)], embs2_normalized[int(j)]) 
                 for i, j in zip(range(len(embs1_normalized)), matched_frames_dtw)]

    # ปรับ threshold โดยใช้ percentile
    threshold = np.percentile(distances, 95)  # เพิ่มเป็น 90 percentile

    total_frames1 = int(len(embs1))
    total_frames2 = int(len(embs2))

    # คำนวณรายละเอียดการจับคู่รายเฟรม
    frame_matching_accuracy = {
        f'Frame {int(i)}': {
            'matched_to_frame': int(matched_idx),
            'distance': float(distance),
            'is_matched': bool(distance <= threshold)
        }
        for i, (matched_idx, distance) in enumerate(zip(matched_frames_dtw, distances))
    }

    # นับจำนวนเฟรมที่ Match
    matched_count = sum(1 for result in frame_matching_accuracy.values() if result['is_matched'])

    # ปรับ weight_factor ให้มีผลกระทบน้อยลง
    weight_factor = 1 / (1 + np.exp((dtw_distance - 200) / 100))

    # คำนวณ Overall Matching Accuracy
    overall_matching_accuracy = (matched_count / min(len(embs1), len(embs2))) * 100 * weight_factor

    end_time = time.time()
    print(f"✅ การทำงาน improved_calculate_matching_accuracy_dtw ใช้เวลา: {end_time - start_time:.4f} วินาที")

    return {
        'frame_matching_details': frame_matching_accuracy,
        'overall_matching_accuracy': overall_matching_accuracy,
        'dtw_distance': float(dtw_distance),
        'total_frames_video1': total_frames1,
        'total_frames_video2': total_frames2,
        'matched_frames': matched_count,
        'dtw_path': path.tolist(),
        'threshold': float(threshold)
    }


REFERENCE_VIDEO_PATH = "Pose1_original_20231124_1334.mp4"
REFERENCE_EMBEDDING_PATH = "embeddings/reference.npy"

def generate_reference_embedding():
    """Generate reference embedding with improved error handling"""
    if not os.path.exists(REFERENCE_VIDEO_PATH):
        print(f"⚠️ Reference video not found: {REFERENCE_VIDEO_PATH}")
        return
    
    model = load_custom_model("model_segment9.keras")
    embedding = process_video_for_web(REFERENCE_VIDEO_PATH, model)
    
    if embedding is None:
        print("❌ Failed to extract embedding from reference video!")
        return

    # Save full sequence of embeddings
    os.makedirs("embeddings", exist_ok=True)
    np.save(REFERENCE_EMBEDDING_PATH, embedding)
    print(f"✅ Reference embedding saved to {REFERENCE_EMBEDDING_PATH}")
    print(f"✅ Embedding shape: {embedding.shape}")


# ✅ เรียกใช้ฟังก์ชันเพื่อสร้าง reference.npy
if __name__ == "__main__":
    generate_reference_embedding()
