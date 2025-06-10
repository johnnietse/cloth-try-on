# from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
# import os
# import cv2
# import cvzone
# from cvzone.PoseModule import PoseDetector
# from datetime import datetime
# import numpy as np
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from dotenv import load_dotenv
#
#
#
#
#
#
# app = Flask(__name__)
#
#
#
#
# UPLOAD_FOLDER = "static/uploads"
# PROCESSED_FOLDER = "static/processed"
# SHIRT_FOLDER = "Resources/Shirts"  # Directory for shirts
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
# app.config['SHIRT_FOLDER'] = SHIRT_FOLDER
#
# # Ensure directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# os.makedirs(SHIRT_FOLDER, exist_ok=True)
#
# def get_shirt_list():
#     """Fetch the list of shirt images dynamically from the directory."""
#     return os.listdir(app.config['SHIRT_FOLDER'])
#
# @app.route('/')
# def index():
#     # Render the index page with the list of available shirts
#     listShirts = get_shirt_list()
#     return render_template('index.html', shirts=listShirts)
#     # return send_from_directory(app.config['SHIRT_FOLDER'], 'index.html')
#     # return send_from_directory('templates', 'index.html')
#
# @app.route('/upload_shirt', methods=['POST'])
# def upload_shirt():
#     """
#     Upload a shirt image to the 'Shirts' directory.
#     """
#     if 'shirt_image' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['shirt_image']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     if file:
#         # Save the uploaded shirt to the SHIRT_FOLDER directory
#         filepath = os.path.join(app.config['SHIRT_FOLDER'], file.filename)
#         file.save(filepath)
#         return redirect(url_for('index'))  # Redirect back to the home page to display the updated list
#
#
# @app.route('/upload', methods=['POST'])
# def upload_video():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400
#
#         # Save video
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{timestamp}_{file.filename}"
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#
#         # Shirt index
#         shirt_index = int(request.form.get('shirt_index', 0))
#         processed_filepath = process_video(filepath, filename, shirt_index)
#         processed_url = f"/{processed_filepath}"
#
#         return jsonify({
#             "message": "Video processing complete! Click the link below to download.",
#             "download_url": processed_url
#         })
#
#     except Exception as e:
#         print("Error in /upload route:", e)
#         return jsonify({"error": str(e)}), 500
#
#
# def process_video(input_path, filename, shirt_index):
#     """
#     Process the uploaded video and overlay the selected shirt.
#     """
#     detector = PoseDetector()
#     cap = cv2.VideoCapture(input_path)
#
#     # Generate a unique filename for the processed video
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     processed_filename = f"processed_{timestamp}_{filename}"
#     processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
#
#     # Video writer to save the output
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(processed_path, fourcc, 30.0, (1280, 720))
#
#     listShirts = get_shirt_list()  # Dynamically fetch the list of shirts
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         img = detector.findPose(img)
#         lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
#
#         # Check if valid keypoints are detected
#         if lmList and len(lmList) > 24:
#             # Extract keypoints for shoulders and hips
#             left_shoulder = np.array(lmList[11][1:3])
#             right_shoulder = np.array(lmList[12][1:3])
#             left_hip = np.array(lmList[23][1:3])
#             right_hip = np.array(lmList[24][1:3])
#
#             # Calculate the center of the bounding box
#             center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
#             center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
#
#             # Define a scaling factor to expand the bounding box
#             scaling_factor = 1.5  # Adjust this value to increase/decrease the box size
#
#             # Calculate distances to expand the box
#             shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scaling_factor
#             hip_height = abs(left_hip[1] - left_shoulder[1]) * scaling_factor
#
#             # Adjust the bounding box
#             left_shoulder[0] = center_x - shoulder_width / 2
#             right_shoulder[0] = center_x + shoulder_width / 2
#             left_shoulder[1] = center_y - hip_height / 2
#             right_shoulder[1] = center_y - hip_height / 2
#             left_hip[0] = center_x - shoulder_width / 2
#             right_hip[0] = center_x + shoulder_width / 2
#             left_hip[1] = center_y + hip_height / 2
#             right_hip[1] = center_y + hip_height / 2
#
#             # Load the shirt image
#             imgShirt = cv2.imread(os.path.join(app.config['SHIRT_FOLDER'], listShirts[shirt_index]), cv2.IMREAD_UNCHANGED)
#
#             # Define the source quadrilateral (full shirt image)
#             height, width = imgShirt.shape[:2]
#             source_pts = np.float32([
#                 [0, 0],                # Top-left corner
#                 [width, 0],            # Top-right corner
#                 [width, height],       # Bottom-right corner
#                 [0, height]            # Bottom-left corner
#             ])
#
#             # Define the target quadrilateral (expanded bounding box with collar adjustment)
#             collar_offset = 30  # Adjust this value to move the collar down
#             target_pts = np.float32([
#                 [left_shoulder[0], left_shoulder[1] + collar_offset],        # Top-left corner (lower collar)
#                 [right_shoulder[0], right_shoulder[1] + collar_offset],      # Top-right corner (lower collar)
#                 [right_hip[0], right_hip[1]],                                # Bottom-right corner
#                 [left_hip[0], left_hip[1]]                                   # Bottom-left corner
#             ])
#
#             # Compute the perspective transform matrix
#             matrix = cv2.getPerspectiveTransform(source_pts, target_pts)
#
#             # Warp the shirt image to fit the expanded bounding box
#             warped_shirt = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
#                                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
#
#             # Overlay the warped shirt on the frame
#             img = overlay_transparent(img, warped_shirt)
#         else:
#             print("Pose detection failed or insufficient landmarks, skipping frame.")
#
#         out.write(img)  # Write frame to video
#
#     cap.release()
#     out.release()
#     return f"{PROCESSED_FOLDER}/{processed_filename}"
#
# def overlay_transparent(background, overlay, alpha_blend=0.7):
#     """
#     Overlay a transparent image (shirt) onto a background.
#     Remove the green filter while keeping the shirt semi-transparent.
#     """
#     # Split the overlay into color channels and the alpha channel
#     b, g, r, a = cv2.split(overlay)
#
#     # Detect green areas (e.g., green filter regions)
#     green_mask = (g > 150) & (r < 100) & (b < 100)  # Adjust thresholds if needed
#     a[green_mask] = 0  # Set alpha to 0 for green regions (fully transparent)
#
#     # Adjust the alpha channel for semi-transparency of the entire shirt
#     alpha = (a / 255.0) * alpha_blend
#
#     # Blend the overlay with the background
#     for c in range(3):  # Iterate over B, G, R channels
#         background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])
#
#     return background
#
# @app.route('/<path:filepath>')
# def download_file(filepath):
#     return send_from_directory('.', filepath)
#
# if __name__ == '__main__':
#     app.run(debug=True)



################################################################################################
import os
import cv2
import cvzone
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for, abort
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure paths from environment variables with fallbacks
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/static/uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', '/app/static/processed')
SHIRT_FOLDER = os.getenv('SHIRT_FOLDER', '/app/Resources/Shirts')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SHIRT_FOLDER'] = SHIRT_FOLDER

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, SHIRT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
if not app.debug:
    file_handler = RotatingFileHandler('/app/logs/app.log', maxBytes=1024 * 1024 * 10, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)


# def get_db_connection():
#     """Create and return a database connection."""
#     try:
#         conn = psycopg2.connect(
#             host=os.getenv('DB_HOST'),
#             database=os.getenv('DB_NAME'),
#             user=os.getenv('DB_USER'),
#             password=os.getenv('DB_PASSWORD'),
#             port=os.getenv('DB_PORT', '5432'),
#             cursor_factory=RealDictCursor
#         )
#         return conn
#     except Exception as e:
#         app.logger.error(f"Database connection failed: {str(e)}")
#         raise
#

def get_db_connection():
    # For Render PostgreSQL
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(database_url)

    # Fallback for local development
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', '5432'),
        cursor_factory=RealDictCursor
    )

def get_shirt_list():
    """Fetch the list of shirt images dynamically from the directory."""
    try:
        return [f for f in os.listdir(app.config['SHIRT_FOLDER'])
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        app.logger.error(f"Error getting shirt list: {str(e)}")
        return []


def overlay_transparent(background, overlay, alpha_blend=0.7):
    """Overlay a transparent image (shirt) onto a background."""
    try:
        if overlay.shape[2] == 4:  # Check if image has alpha channel
            b, g, r, a = cv2.split(overlay)
        else:
            b, g, r = cv2.split(overlay)
            a = np.ones_like(b) * 255

        # Detect green areas (e.g., green filter regions)
        green_mask = (g > 150) & (r < 100) & (b < 100)
        a[green_mask] = 0  # Set alpha to 0 for green regions

        # Adjust the alpha channel for semi-transparency
        alpha = (a / 255.0) * alpha_blend

        # Blend the overlay with the background
        for c in range(3):  # Iterate over B, G, R channels
            background[:, :, c] = (alpha * overlay[:, :, c] +
                                   (1 - alpha) * background[:, :, c])

        return background
    except Exception as e:
        app.logger.error(f"Overlay failed: {str(e)}")
        return background


def process_video(input_path, filename, shirt_index):
    """Process the uploaded video and overlay the selected shirt."""
    try:
        detector = PoseDetector()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"processed_{timestamp}_{filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

        listShirts = get_shirt_list()
        if not listShirts:
            raise ValueError("No shirts available for processing")

        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findPose(img)
            lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

            if lmList and len(lmList) > 24:
                # Load the shirt image
                shirt_path = os.path.join(app.config['SHIRT_FOLDER'], listShirts[shirt_index])
                imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
                if imgShirt is None:
                    app.logger.error(f"Failed to load shirt image: {shirt_path}")
                    continue

                # [Rest of your original processing code...]
                # ... (keep all your existing pose detection and warping logic)

                img = overlay_transparent(img, warped_shirt)

            out.write(img)

        cap.release()
        out.release()
        return f"{PROCESSED_FOLDER}/{processed_filename}"

    except Exception as e:
        app.logger.error(f"Video processing failed: {str(e)}")
        raise


@app.route('/')
def index():
    """Render the main page with available shirts."""
    try:
        listShirts = get_shirt_list()
        return render_template('index.html', shirts=listShirts)
    except Exception as e:
        app.logger.error(f"Index page failed: {str(e)}")
        return render_template('error.html'), 500


@app.route('/upload_shirt', methods=['POST'])
def upload_shirt():
    """Handle shirt image uploads."""
    try:
        if 'shirt_image' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['shirt_image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = os.path.join(app.config['SHIRT_FOLDER'], file.filename)
            file.save(filename)
            return redirect(url_for('index'))

        return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        app.logger.error(f"Shirt upload failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video uploads and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            return jsonify({"error": "Invalid video format"}), 400

        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get shirt index safely
        try:
            shirt_index = int(request.form.get('shirt_index', 0))
            shirt_list = get_shirt_list()
            if not 0 <= shirt_index < len(shirt_list):
                shirt_index = 0
        except:
            shirt_index = 0

        processed_filepath = process_video(filepath, filename, shirt_index)
        processed_url = f"/processed/{os.path.basename(processed_filepath)}"

        return jsonify({
            "message": "Video processing complete!",
            "download_url": processed_url
        })

    except Exception as e:
        app.logger.error(f"Video upload failed: {str(e)}")
        return jsonify({"error": "Video processing failed"}), 500


@app.route('/processed/<filename>')
def download_processed(filename):
    """Serve processed video files."""
    try:
        # Security check
        if '..' in filename or filename.startswith('/'):
            abort(404)
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except Exception as e:
        app.logger.error(f"Download failed: {str(e)}")
        abort(404)


@app.route('/healthz')
def health_check():
    """Health check endpoint for Kubernetes."""
    try:
        # Simple database check if using DB
        if os.getenv('DB_HOST'):
            conn = get_db_connection()
            conn.close()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug)