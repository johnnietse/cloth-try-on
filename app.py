import os
import cv2
import cvzone
import numpy as np
import requests
import base64
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for, abort, session
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

# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')

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

# API Keys (Set these in your environment variables)
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
INSTAGRAM_APP_ID = os.environ.get('INSTAGRAM_APP_ID')
INSTAGRAM_APP_SECRET = os.environ.get('INSTAGRAM_APP_SECRET')
INSTAGRAM_REDIRECT_URI = os.environ.get('INSTAGRAM_REDIRECT_URI', 'http://localhost:5000/instagram_callback')

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


# New routes for Cohere, DALLE-3 and Instagram
@app.route('/cohere_chat', methods=['POST'])
def cohere_chat():
    try:
        data = request.json
        user_message = data.get('message')
        conversation = data.get('conversation', [])

        headers = {
            'Authorization': f'Bearer {COHERE_API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "command",
            "message": user_message,
            "chat_history": conversation,
            "prompt_truncation": "AUTO",
            "temperature": 0.3
        }

        response = requests.post(
            'https://api.cohere.ai/v1/chat',
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            app.logger.error(f"Cohere API error: {response.text}")
            return jsonify({"error": "Failed to get response from Cohere"}), 500

        cohere_data = response.json()
        return jsonify({
            "text": cohere_data.get('text', ''),
            "conversation_id": cohere_data.get('conversation_id', '')
        })
    except Exception as e:
        app.logger.error(f"cohere_chat failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/instagram_auth')
def instagram_auth():
    # Start Instagram OAuth flow
    auth_url = (
        f"https://api.instagram.com/oauth/authorize?"
        f"client_id={INSTAGRAM_APP_ID}&"
        f"redirect_uri={INSTAGRAM_REDIRECT_URI}&"
        "scope=user_profile,user_media&"
        "response_type=code"
    )
    return redirect(auth_url)


@app.route('/instagram_callback')
def instagram_callback():
    code = request.args.get('code')
    if not code:
        return jsonify({"error": "Authorization failed"}), 400

    # Exchange code for access token
    token_data = {
        'client_id': INSTAGRAM_APP_ID,
        'client_secret': INSTAGRAM_APP_SECRET,
        'grant_type': 'authorization_code',
        'redirect_uri': INSTAGRAM_REDIRECT_URI,
        'code': code
    }

    response = requests.post(
        'https://api.instagram.com/oauth/access_token',
        data=token_data
    )

    if response.status_code != 200:
        app.logger.error(f"Instagram token exchange failed: {response.text}")
        return jsonify({"error": "Failed to get access token"}), 500

    token_info = response.json()
    access_token = token_info.get('access_token')
    user_id = token_info.get('user_id')

    # Store token in session
    session['instagram_token'] = access_token
    session['instagram_user_id'] = user_id

    return redirect(url_for('index'))


@app.route('/post_to_instagram', methods=['POST'])
def post_to_instagram():
    try:
        data = request.json
        image_url = data.get('image_url')

        # Check if user is authenticated
        access_token = session.get('instagram_token')
        user_id = session.get('instagram_user_id')

        if not access_token or not user_id:
            return jsonify({"error": "Not authenticated with Instagram"}), 401

        # Get absolute path to image
        if not image_url.startswith('/static/processed/'):
            return jsonify({"error": "Invalid image path"}), 400

        image_path = os.path.join(BASE_DIR, image_url[1:])

        # Step 1: Create media container
        container_url = f"https://graph.facebook.com/v18.0/{user_id}/media"
        container_params = {
            'image_url': request.host_url + image_url.lstrip('/'),
            'caption': 'Created with Virtual Try-On #VirtualTryOn #FashionTech',
            'access_token': access_token
        }

        container_resp = requests.post(container_url, params=container_params)
        if container_resp.status_code != 200:
            app.logger.error(f"Instagram container error: {container_resp.text}")
            return jsonify({"error": "Failed to create media container"}), 500

        container_id = container_resp.json().get('id')

        # Step 2: Publish the container
        publish_url = f"https://graph.facebook.com/v18.0/{user_id}/media_publish"
        publish_params = {
            'creation_id': container_id,
            'access_token': access_token
        }

        publish_resp = requests.post(publish_url, params=publish_params)
        if publish_resp.status_code != 200:
            app.logger.error(f"Instagram publish error: {publish_resp.text}")
            return jsonify({"error": "Failed to publish media"}), 500

        return jsonify({"success": True, "post_id": publish_resp.json().get('id')})
    except Exception as e:
        app.logger.error(f"post_to_instagram failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug)
