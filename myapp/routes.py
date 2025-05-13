# from flask import Blueprint, redirect, url_for
#
# from .extensions import db
# from .models import User
#
# main = Blueprint('main', __name__)
#
# @main.route('/')
# def index():
#     users = User.query.all()
#     users_list_html = [f"<li>{ user.username }</li>" for user in users]
#     return f"<ul>{''.join(users_list_html)}</ul>"
#
# @main.route('/add/<username>')
# def add_user(username):
#     db.session.add(User(username=username))
#     db.session.commit()
#     return redirect(url_for("main.index"))


from flask import Blueprint, request, render_template, send_from_directory, jsonify, redirect, url_for
from .models import VideoUpload
from .extensions import db
import os
import cv2
import numpy as np
from datetime import datetime
from cvzone.PoseModule import PoseDetector

main = Blueprint('main', __name__)

def get_shirt_list():
    """Fetch the list of shirt images dynamically from the directory."""
    return os.listdir(main.app.config['SHIRT_FOLDER'])

@main.route('/')
def index():
    listShirts = get_shirt_list()
    return render_template('index.html', shirts=listShirts)

@main.route('/upload_shirt', methods=['POST'])
def upload_shirt():
    if 'shirt_image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['shirt_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(main.app.config['SHIRT_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('main.index'))

@main.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(main.app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        shirt_index = int(request.form.get('shirt_index', 0))
        processed_filepath = process_video(filepath, filename, shirt_index)
        processed_url = f"/{processed_filepath}"

        # Save to database
        video = VideoUpload(
            original_filename=filename,
            processed_filename=processed_filepath,
            shirt_index=shirt_index
        )
        db.session.add(video)
        db.session.commit()

        return jsonify({
            "message": "Video processing complete! Click the link below to download.",
            "download_url": processed_url
        })

    except Exception as e:
        print("Error in /upload route:", e)
        return jsonify({"error": str(e)}), 500

def process_video(input_path, filename, shirt_index):
    detector = PoseDetector()
    cap = cv2.VideoCapture(input_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"processed_{timestamp}_{filename}"
    processed_path = os.path.join(main.app.config['PROCESSED_FOLDER'], processed_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, 30.0, (1280, 720))

    listShirts = get_shirt_list()
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList and len(lmList) > 24:
            left_shoulder = np.array(lmList[11][1:3])
            right_shoulder = np.array(lmList[12][1:3])
            left_hip = np.array(lmList[23][1:3])
            right_hip = np.array(lmList[24][1:3])

            center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
            center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4

            scaling_factor = 1.5
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scaling_factor
            hip_height = abs(left_hip[1] - left_shoulder[1]) * scaling_factor

            left_shoulder[0] = center_x - shoulder_width / 2
            right_shoulder[0] = center_x + shoulder_width / 2
            left_shoulder[1] = center_y - hip_height / 2
            right_shoulder[1] = center_y - hip_height / 2
            left_hip[0] = center_x - shoulder_width / 2
            right_hip[0] = center_x + shoulder_width / 2
            left_hip[1] = center_y + hip_height / 2
            right_hip[1] = center_y + hip_height / 2

            imgShirt = cv2.imread(os.path.join(main.app.config['SHIRT_FOLDER'], listShirts[shirt_index]), cv2.IMREAD_UNCHANGED)

            height, width = imgShirt.shape[:2]
            source_pts = np.float32([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ])

            collar_offset = 30
            target_pts = np.float32([
                [left_shoulder[0], left_shoulder[1] + collar_offset],
                [right_shoulder[0], right_shoulder[1] + collar_offset],
                [right_hip[0], right_hip[1]],
                [left_hip[0], left_hip[1]]
            ])

            matrix = cv2.getPerspectiveTransform(source_pts, target_pts)
            warped_shirt = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            img = overlay_transparent(img, warped_shirt)

        out.write(img)

    cap.release()
    out.release()
    return f"static/processed/{processed_filename}"

def overlay_transparent(background, overlay, alpha_blend=0.7):
    b, g, r, a = cv2.split(overlay)
    green_mask = (g > 150) & (r < 100) & (b < 100)
    a[green_mask] = 0
    alpha = (a / 255.0) * alpha_blend
    for c in range(3):
        background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])
    return background

@main.route('/<path:filepath>')
def download_file(filepath):
    return send_from_directory('..', filepath)