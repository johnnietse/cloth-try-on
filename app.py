import os
import cv2
import cvzone
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for, abort
from cvzone.PoseModule import PoseDetector
import logging
from logging.handlers import RotatingFileHandler
import shutil


# OpenCV compatibility workaround
np.int = int
np.float = float
np.bool = bool

if not hasattr(cv2.dnn, 'DictValue'):
    cv2.dnn.DictValue = type('DictValue', (), {})

app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
SHIRT_FOLDER = os.path.join(BASE_DIR, 'static', 'shirts')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SHIRT_FOLDER'] = SHIRT_FOLDER

# Create directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, SHIRT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
if not app.debug:
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)


def get_shirt_list():
    try:
        return [f for f in os.listdir(app.config['SHIRT_FOLDER'])
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        app.logger.error(f"get_shirt_list failed: {e}")
        return []


@app.route('/delete_shirt/<filename>', methods=['DELETE'])
def delete_shirt(filename):
    try:
        # Secure the filename
        if '..' in filename or filename.startswith('/'):
            return jsonify({"error": "Invalid filename"}), 400
        
        shirt_path = os.path.join(app.config['SHIRT_FOLDER'], filename)
        
        # Create backup before deleting
        backup_dir = os.path.join(BASE_DIR, 'static', 'deleted_shirts')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        
        if os.path.exists(shirt_path):
            # Move to backup instead of permanent delete
            shutil.move(shirt_path, backup_path)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        app.logger.error(f"delete_shirt failed: {e}")
        return jsonify({"error": "Internal server error"}), 500



def overlay_transparent(background, overlay, alpha_blend=0.7):
    try:
        if overlay.shape[2] == 4:
            b, g, r, a = cv2.split(overlay)
        else:
            b, g, r = cv2.split(overlay)
            a = np.ones_like(b) * 255

        green_mask = (g > 150) & (r < 100) & (b < 100)
        a[green_mask] = 0

        alpha = (a / 255.0) * alpha_blend
        for c in range(3):
            background[:, :, c] = (alpha * overlay[:, :, c] +
                                   (1 - alpha) * background[:, :, c])

        return background
    except Exception as e:
        app.logger.error(f"overlay_transparent failed: {e}")
        return background


def process_image(user_image_path, shirt_index):
    detector = PoseDetector()
    img = cv2.imread(user_image_path)
    if img is None:
        raise ValueError("Failed to load user image")

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    shirts = get_shirt_list()
    if not shirts:
        raise ValueError("No shirt images found")

    if lmList and len(lmList) > 24:
        shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
        imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if imgShirt is None:
            raise ValueError(f"Failed to load shirt image: {shirt_path}")

        # Extract keypoints
        left_shoulder = np.array(lmList[11][1:3])
        right_shoulder = np.array(lmList[12][1:3])
        left_hip = np.array(lmList[23][1:3])
        right_hip = np.array(lmList[24][1:3])

        # Calculate center
        center_x = np.mean([left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]])
        center_y = np.mean([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]])
        scale = 1.5

        # Calculate dimensions
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scale
        hip_height = abs(left_hip[1] - left_shoulder[1]) * scale

        # Define points
        left_shoulder = [center_x - shoulder_width / 2, center_y - hip_height / 2]
        right_shoulder = [center_x + shoulder_width / 2, center_y - hip_height / 2]
        left_hip = [center_x - shoulder_width / 2, center_y + hip_height / 2]
        right_hip = [center_x + shoulder_width / 2, center_y + hip_height / 2]

        # Perspective transformation
        src_pts = np.float32([
            [0, 0],
            [imgShirt.shape[1], 0],
            [imgShirt.shape[1], imgShirt.shape[0]],
            [0, imgShirt.shape[0]]
        ])

        dst_pts = np.float32([
            [left_shoulder[0], left_shoulder[1] + 30],
            [right_shoulder[0], right_shoulder[1] + 30],
            [right_hip[0], right_hip[1]],
            [left_hip[0], left_hip[1]]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
                                     borderValue=(0, 0, 0, 0))

        # Overlay shirt on image
        result = overlay_transparent(img, warped)

        # Save processed image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"processed_{timestamp}.jpg"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, result)

        return processed_filename
    else:
        raise ValueError("Pose detection failed - could not find body landmarks")


@app.route('/')
def index():
    shirts = get_shirt_list()
    return render_template('index.html', shirts=shirts)


@app.route('/upload_shirt', methods=['POST'])
def upload_shirt():
    try:
        file = request.files.get('shirt_image')
        if not file or file.filename == '':
            return jsonify({"error": "No shirt image uploaded"}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file type"}), 400

        filename = os.path.join(app.config['SHIRT_FOLDER'], file.filename)
        file.save(filename)
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"upload_shirt failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Get user image
        user_image = request.files.get('user_image')
        if not user_image or user_image.filename == '':
            return jsonify({"error": "No user image uploaded"}), 400

        if not user_image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file type"}), 400

        # Save user image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_filename = f"user_{timestamp}_{user_image.filename}"
        user_path = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
        user_image.save(user_path)

        # Get shirt index
        shirt_index = int(request.form.get('shirt_index', 0))
        shirt_list = get_shirt_list()
        if not 0 <= shirt_index < len(shirt_list):
            shirt_index = 0

        # Process image
        processed_filename = process_image(user_path, shirt_index)
        processed_url = f"/static/processed/{processed_filename}"

        return jsonify({
            "success": True,
            "message": "Image processed successfully!",
            "processed_url": processed_url
        })
    except Exception as e:
        app.logger.error(f"upload_image failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug)
