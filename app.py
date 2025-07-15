import os
import cv2
import cvzone
import numpy as np
import requests
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for, send_file, abort
from cvzone.PoseModule import PoseDetector
import logging
from logging.handlers import RotatingFileHandler
import shutil
import pkg_resources
from flask.sessions import SecureCookieSessionInterface


# OpenCV compatibility workaround
np.int = int
np.float = float
np.bool = bool

app = Flask(__name__)

# Dependency version check
required = {
    'Flask': '2.0.1',
    'opencv-python-headless': '4.5.4.60',
    'numpy': '1.21.6',
    'protobuf': '3.20.3'
}

for package, version in required.items():
    try:
        installed_version = pkg_resources.get_distribution(package).version
        if installed_version != version:
            raise ImportError(f"{package} version mismatch. Required {version}, found {installed_version}")
    except pkg_resources.DistributionNotFound:
        raise ImportError(f"{package} is not installed")

# Custom session interface
class CustomSessionInterface(SecureCookieSessionInterface):
    def save_session(self, *args, **kwargs):
        return super(CustomSessionInterface, self).save_session(*args, **kwargs)

app.session_interface = CustomSessionInterface()

# Set secret key
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

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

# API Keys
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

def get_shirt_list():
    try:
        return [f for f in os.listdir(app.config['SHIRT_FOLDER'])
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        app.logger.error(f"get_shirt_list failed: {e}")
        return []

def overlay_transparent(background, overlay, alpha_blend=0.7):
    try:
        if overlay.shape[2] == 4:
            b, g, r, a = cv2.split(overlay)
        else:
            b, g, r = cv2.split(overlay)
            a = np.ones_like(b) * 255

        green_mask = (g > 150) & (r < 100) & (b < 100)
        a[green_mask] = 0

        alpha = (a / 255.0)*alpha_blend

        for c in range(3):
            background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])

        return background
    except Exception as e:
        app.logger.error(f"overlay_transparent failed: {e}")
        return background

def process_image(user_image_path, shirt_index):
    detector = PoseDetector()
    img = cv2.imread(user_image_path)
    if img is None:
        raise ValueError("Failed to load user image")

    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    shirts = get_shirt_list()
    if not shirts:
        raise ValueError("No shirt images found")

    if lmList and len(lmList) > 24:
        shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
        imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if imgShirt is None:
            raise ValueError(f"Failed to load shirt image: {shirt_path}")

        # Get original landmarks
        left_shoulder = np.array(lmList[11][1:3])
        right_shoulder = np.array(lmList[12][1:3])
        left_hip = np.array(lmList[23][1:3])
        right_hip = np.array(lmList[24][1:3])

        # Calculate center point
        center_x = np.mean([left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]])
        center_y = np.mean([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]])
        
        # Calculate dimensions with scale factor
        scale = 1.5
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])*scale
        hip_height = abs(left_hip[1] - left_shoulder[1])*scale

        # Create synthetic torso points
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
        
        # Apply vertical offset to shoulders (+30px)
        dst_pts = np.float32([
            [left_shoulder[0], left_shoulder[1] + 30], 
            [right_shoulder[0], right_shoulder[1] + 30], 
            [right_hip[0], right_hip[1]], 
            [left_hip[0], left_hip[1]] 
        ])

        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0, 0))
        
        # Overlay with transparency
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

# Privacy and legal pages
@app.route('/privacy')
def privacy():
    return render_template('privacy.html'), 200, {'Link': '<https://virtual-try-on-yg9v.onrender.com/privacy>; rel="canonical"'}

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/data_deletion')
def data_deletion():
    return render_template('data_deletion.html')

# Static files
@app.route('/robots.txt')
def robots():
    return send_from_directory(app.template_folder, 'robots.txt')

@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(app.template_folder, 'sitemap.xml')

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


# @app.route('/get_processed_image/<filename>')
# def get_processed_image(filename):
#     try:
#         # Secure filename input
#         if '..' in filename or '/' in filename:
#             abort(404)
        
#         image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
#         if not os.path.exists(image_path):
#             abort(404)
        
#         # Set proper headers for image sharing
#         response = send_file(image_path, mimetype='image/jpeg')
#         response.headers['Access-Control-Allow-Origin'] = '*'
#         return response
#     except Exception as e:
#         app.logger.error(f"Error serving processed image: {e}")
#         abort(500)



@app.route('/get_processed_image/<filename>')
def get_processed_image(filename):
    try:
        # Secure filename input
        if '..' in filename or '/' in filename:
            abort(404)
        
        image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
        if not os.path.exists(image_path):
            abort(404)
        
        # Set proper headers for image sharing
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        app.logger.error(f"Error serving processed image: {e}")
        abort(500)


# Health checks
@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy", "time": datetime.utcnow().isoformat()}), 200

@app.route('/dependency_check')
def dependency_check():
    dependencies = {
        'Flask': '2.0.1',
        'opencv-python-headless': '4.5.4.60',
        'numpy': '1.21.6',
        'protobuf': '3.20.3',
        'cvzone': '1.5.6'
    }
    
    status = {}
    for package, expected in dependencies.items():
        try:
            actual = pkg_resources.get_distribution(package).version
            status[package] = {
                'expected': expected,
                'actual': actual,
                'match': actual == expected
            }
        except Exception as e:
            status[package] = {
                'error': str(e)
            }
    
    return jsonify(status)

# Error handling
@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal Server Error: {e}")
    return render_template('500.html'), 500

# Main entry point
if __name__ == '__main__':
    # Verify dependencies
    try:
        import cv2
        import cvzone
        app.logger.info(f"OpenCV version: {cv2.__version__}")
        app.logger.info(f"CVZone version: {cvzone.__version__}")
    except ImportError as e:
        app.logger.error(f"Import error: {e}")
    
    # Start the app
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    app.logger.info(f"Starting app on port {port} in {'debug' if debug else 'production'} mode")
    app.run(host='0.0.0.0', port=port, debug=debug)
