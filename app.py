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

        # (Optional) Disable green-screen removal for testing
        green_mask = (g > 150) & (r < 100) & (b < 100)
        a[green_mask] = 0

        alpha = (a / 255.0) * alpha_blend

        for c in range(3):
            background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])

        return background
    except Exception as e:
        app.logger.error(f"overlay_transparent failed: {e}")
        return background


# def overlay_transparent(background, overlay, alpha_blend=0.7):
#     try:
#         if overlay.shape[2] == 4:
#             b, g, r, a = cv2.split(overlay)
#         else:
#             b, g, r = cv2.split(overlay)
#             a = np.ones_like(b) * 255

#         # # Improved green screen 
#         # green_mask = (g > 150) & (r < 100) & (b < 100)
#         # a[green_mask] = 0


#         alpha = (a / 255.0) * alpha_blend
#         for c in range(3):
#             background[:, :, c] = (alpha * overlay[:, :, c] +
#                                    (1 - alpha) * background[:, :, c])

#         return background
#     except Exception as e:
#         app.logger.error(f"overlay_transparent failed: {e}")
#         return background

        # Create mask from alpha channel
        # mask = a > 0


        # # Create mask and inverse mask
        # alpha = a.astype(float) / 255
        # alpha = alpha * alpha_blend
        # alpha_inv = 1.0 - alpha

        # # alpha = (a / 255.0) * alpha_blend
        
        # for c in range(3):
        #     background[:, :, c] = (alpha * overlay[:, :, c] +
        #                            (1 - alpha) * background[:, :, c])

        # Apply overlay using mask
    #     for c in range(3):
    #         background[:, :, c] = np.where(
    #             mask,
    #             overlay[:, :, c] * alpha_blend + background[:, :, c] * (1 - alpha_blend),
    #             background[:, :, c]
    #         )

    #     return background
    # except Exception as e:
    #     app.logger.error(f"overlay_transparent failed: {e}")
    #     return background


# def process_image(user_image_path, shirt_index):
#     detector = PoseDetector()
#     img = cv2.imread(user_image_path)
#     if img is None:
#         raise ValueError("Failed to load user image")

#     app.logger.info(f"User image loaded: {user_image_path}")


#     img = detector.findPose(img)
#     lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

#     app.logger.info(f"Pose landmarks: {lmList}")

#     shirts = get_shirt_list()
#     if not shirts:
#         raise ValueError("No shirt images found")

#     if lmList and len(lmList) > 24:
#         shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
#         imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
#         if imgShirt is None:
#             raise ValueError(f"Failed to load shirt image: {shirt_path}")

#         app.logger.info(f"Shirt loaded: {shirt_path}")
#         cv2.imwrite("debug_shirt_raw.png", imgShirt)

#         # # Extract keypoints
#         # left_shoulder = np.array(lmList[11][1:3])
#         # right_shoulder = np.array(lmList[12][1:3])
#         # left_hip = np.array(lmList[23][1:3])
#         # right_hip = np.array(lmList[24][1:3])


#         # # Calculate torso dimensions with better scaling
#         # torso_width = np.linalg.norm(left_shoulder - right_shoulder) 
#         # torso_height = np.linalg.norm(left_shoulder - left_hip) 

#         # # Calculate shirt position with scaling factors
#         # scale_width = 1.5
#         # scale_height = 1.8
        
#         # # # Calculate center
#         # # center_x = np.mean([left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]])
#         # # center_y = np.mean([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]])
#         # # scale = 1.5

#         # # # Calculate dimensions
#         # # shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scale
#         # # hip_height = abs(left_hip[1] - left_shoulder[1]) * scale

#         # # # Define points
#         # # left_shoulder = [center_x - shoulder_width / 2, center_y - hip_height / 2]
#         # # right_shoulder = [center_x + shoulder_width / 2, center_y - hip_height / 2]
#         # # left_hip = [center_x - shoulder_width / 2, center_y + hip_height / 2]
#         # # right_hip = [center_x + shoulder_width / 2, center_y + hip_height / 2]

#         # # # Perspective transformation
#         # # src_pts = np.float32([
#         # #     [0, 0],
#         # #     [imgShirt.shape[1], 0],
#         # #     [imgShirt.shape[1], imgShirt.shape[0]],
#         # #     [0, imgShirt.shape[0]]
#         # # ])

#         # # dst_pts = np.float32([
#         # #     [left_shoulder[0], left_shoulder[1] + 30],
#         # #     [right_shoulder[0], right_shoulder[1] + 30],
#         # #     [right_hip[0], right_hip[1]],
#         # #     [left_hip[0], left_hip[1]]
#         # # ])

#         # # # Calculate shirt position
#         # # center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         # # center_y = int((left_shoulder[1] + left_hip[1]) / 2)
        
#         # # # Define points with better positioning
#         # # shirt_top_left = [int(center_x - torso_width/2), int(center_y - torso_height/2)]
#         # # shirt_top_right = [int(center_x + torso_width/2), int(center_y - torso_height/2)]
#         # # shirt_bottom_right = [int(center_x + torso_width/2), int(center_y + torso_height/2)]
#         # # shirt_bottom_left = [int(center_x - torso_width/2), int(center_y + torso_height/2)]

#         # # # Perspective transformation
#         # # src_pts = np.float32([
#         # #     [0, 0],
#         # #     [imgShirt.shape[1], 0],
#         # #     [imgShirt.shape[1], imgShirt.shape[0]],
#         # #     [0, imgShirt.shape[0]]
#         # # ])

#         # # dst_pts = np.float32([
#         # #     shirt_top_left,
#         # #     shirt_top_right,
#         # #     shirt_bottom_right,
#         # #     shirt_bottom_left
#         # # ])

#         # # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
#         # # warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
#         # #                              borderValue=(0, 0, 0, 0))


#         # # Define shirt placement points
#         # shirt_top_left = [
#         #     int((left_shoulder[0] + right_shoulder[0])/2 - (torso_width * scale_width)/2),
#         #     int(left_shoulder[1] - torso_height * 0.1)
#         # ]
        
#         # shirt_top_right = [
#         #     int((left_shoulder[0] + right_shoulder[0])/2 + (torso_width * scale_width)/2),
#         #     int(right_shoulder[1] - torso_height * 0.1)
#         # ]
        
#         # shirt_bottom_right = [
#         #     int(right_hip[0] + torso_width * 0.1),
#         #     int(right_hip[1] - torso_height * 0.1)
#         # ]
        
#         # shirt_bottom_left = [
#         #     int(left_hip[0] - torso_width * 0.1),
#         #     int(left_hip[1] - torso_height * 0.1)
#         # ]

#         # # Debug: Print points
#         # app.logger.info(f"Shirt points: TL:{shirt_top_left}, TR:{shirt_top_right}, BR:{shirt_bottom_right}, BL:{shirt_bottom_left}")

#         # # Perspective transformation
#         # src_pts = np.float32([
#         #     [0, 0],
#         #     [imgShirt.shape[1], 0],
#         #     [imgShirt.shape[1], imgShirt.shape[0]],
#         #     [0, imgShirt.shape[0]]
#         # ])

#         # dst_pts = np.float32([
#         #     shirt_top_left,
#         #     shirt_top_right,
#         #     shirt_bottom_right,
#         #     shirt_bottom_left
#         # ])

#         # # Calculate perspective transform
#         # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
#         # warped = cv2.warpPerspective(
#         #     imgShirt, matrix, (img.shape[1], img.shape[0]),
#         #     borderMode=cv2.BORDER_CONSTANT,
#         #     borderValue=(0, 0, 0, 0)
#         # )


#         # Extract keypoints
#         left_shoulder = np.array(lmList[11][1:3])
#         right_shoulder = np.array(lmList[12][1:3])
#         left_hip = np.array(lmList[23][1:3])
#         right_hip = np.array(lmList[24][1:3])

#         img_height, img_width = img.shape[:2]

#         # # Calculate center
#         # center_x = np.mean([left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]])
#         # center_y = np.mean([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]])
#         # scale = 1.5

#         # # Calculate dimensions
#         # shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scale
#         # hip_height = abs(left_hip[1] - left_shoulder[1]) * scale

#         # # Define points
#         # left_shoulder = [center_x - shoulder_width / 2, center_y - hip_height / 2]
#         # right_shoulder = [center_x + shoulder_width / 2, center_y - hip_height / 2]
#         # left_hip = [center_x - shoulder_width / 2, center_y + hip_height / 2]
#         # right_hip = [center_x + shoulder_width / 2, center_y + hip_height / 2]

#         # app.logger.info(f"Overlay points: {left_shoulder}, {right_shoulder}, {left_hip}, {right_hip}")

#         # Calculate the top and bottom bounds of the torso
#         top_y = int(min(left_shoulder[1], right_shoulder[1]))
#         bottom_y = int(max(left_hip[1], right_hip[1]))
#         torso_height = bottom_y - top_y
#         torso_width = int(np.linalg.norm(left_shoulder - right_shoulder))

#         # Add padding to width/height
#         pad_width = int(torso_width * 0.5)
#         pad_height = int(torso_height * 0.2)

#         # Define four destination points (bounding box with padding)
#         x1 = max(0, int(left_shoulder[0]) - pad_width)
#         y1 = max(0, top_y - pad_height)
#         x2 = min(img_width - 1, int(right_shoulder[0]) + pad_width)
#         y2 = min(img_height - 1, bottom_y + pad_height)

#         # # Perspective transformation
#         # src_pts = np.float32([
#         #     [0, 0],
#         #     [imgShirt.shape[1], 0],
#         #     [imgShirt.shape[1], imgShirt.shape[0]],
#         #     [0, imgShirt.shape[0]]
#         # ])

#         # dst_pts = np.float32([
#         #     [left_shoulder[0], left_shoulder[1] + 30],
#         #     [right_shoulder[0], right_shoulder[1] + 30],
#         #     [right_hip[0], right_hip[1]],
#         #     [left_hip[0], left_hip[1]]
#         # ])

#         dst_pts = np.float32([
#             [x1, y1],
#             [x2, y1],
#             [x2, y2],
#             [x1, y2]
#         ])

#         src_pts = np.float32([
#             [0, 0],
#             [imgShirt.shape[1], 0],
#             [imgShirt.shape[1], imgShirt.shape[0]],
#             [0, imgShirt.shape[0]]
#         ])

#         app.logger.info(f"Overlay points: {dst_pts.tolist()}")

#         # Visualize placement points (optional)
#         for pt in dst_pts:
#             cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
#         cv2.imwrite("debug_pose_overlay_points.jpg", img)
        
#         # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
#         # warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
#         #                              borderValue=(0, 0, 0, 0))

#         matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
#         warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
#                                      borderValue=(0, 0, 0, 0))


        
#         cv2.imwrite("debug_warped.png", warped)

        
#         # Overlay shirt on image
#         # result = overlay_transparent(img, warped)
#         result = overlay_transparent(img, warped, alpha_blend=1.0)

#         cv2.imwrite("debug_overlay_result.png", result)


#         # Save processed image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         processed_filename = f"processed_{timestamp}.jpg"
#         processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
#         cv2.imwrite(processed_path, result)

#         return processed_filename
#     else:
#         raise ValueError("Pose detection failed - could not find body landmarks")

# def process_image(user_image_path, shirt_index):
#     detector = PoseDetector()
#     img = cv2.imread(user_image_path)
#     if img is None:
#         raise ValueError("Failed to load user image")

#     # app.logger.info(f"User image loaded: {user_image_path}")

#     img = detector.findPose(img)
#     lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
#     app.logger.info(f"Pose landmarks: {lmList}")

#     shirts = get_shirt_list()
#     if not shirts:
#         raise ValueError("No shirt images found")

#     if lmList and len(lmList) > 24:
#         shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
#         imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
#         if imgShirt is None or imgShirt.shape[2] != 4:
#             raise ValueError(f"Failed to load valid shirt image with alpha channel: {shirt_path}")

#         app.logger.info(f"Shirt loaded: {shirt_path}")
#         cv2.imwrite("debug_shirt_raw.png", imgShirt)

#         # Extract keypoints
#         left_shoulder = np.array(lmList[11][1:3])
#         right_shoulder = np.array(lmList[12][1:3])
#         left_hip = np.array(lmList[23][1:3])
#         right_hip = np.array(lmList[24][1:3])

#         img_height, img_width = img.shape[:2]

#         # Top/bottom boundaries
#         top_y = int(min(left_shoulder[1], right_shoulder[1]))
#         bottom_y = int(max(left_hip[1], right_hip[1]))
#         torso_height = bottom_y - top_y
#         torso_width = int(np.linalg.norm(left_shoulder - right_shoulder))

#         # Pad and clamp
#         pad_width = int(torso_width * 0.5)
#         pad_height = max(50, int(torso_height * 0.2))  # Ensure shirt has height

#         x1 = max(0, int(left_shoulder[0]) - pad_width)
#         x2 = min(img_width - 1, int(right_shoulder[0]) + pad_width)
#         y1 = max(0, top_y - pad_height)
#         y2 = min(img_height - 1, bottom_y + pad_height)

#         if abs(y2 - y1) < 50:
#             y2 = y1 + 50  # Prevent flat bounding box

#         dst_pts = np.float32([
#             [x1, y1],
#             [x2, y1],
#             [x2, y2],
#             [x1, y2]
#         ])
#         src_pts = np.float32([
#             [0, 0],
#             [imgShirt.shape[1], 0],
#             [imgShirt.shape[1], imgShirt.shape[0]],
#             [0, imgShirt.shape[0]]
#         ])

#         app.logger.info(f"Overlay points: {dst_pts.tolist()}")

#         # Debug pose box
#         for pt in dst_pts:
#             cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
#         cv2.imwrite("debug_pose_overlay_points.jpg", img)

#         # Warp shirt
#         warped = cv2.warpPerspective(imgShirt, cv2.getPerspectiveTransform(src_pts, dst_pts),
#                                      (img.shape[1], img.shape[0]), borderValue=(0, 0, 0, 0))
#         if warped.shape[2] != 4:
#             warped = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)

#         cv2.imwrite("debug_warped.png", warped)

#         # Overlay shirt
#         result = overlay_transparent(img, warped, alpha_blend=1.0)
#         cv2.imwrite("debug_overlay_result.jpg", result)

#         # Save result
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         processed_filename = f"processed_{timestamp}.jpg"
#         processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
#         cv2.imwrite(processed_path, result)

#         return processed_filename
#     else:
#         raise ValueError("Pose detection failed - could not find body landmarks")

# def process_image(user_image_path, shirt_index):
#     detector = PoseDetector()
#     img = cv2.imread(user_image_path)
#     if img is None:
#         raise ValueError("Failed to load user image")

#     # Create a copy for pose detection
#     pose_img = img.copy()
#     pose_img = detector.findPose(pose_img)
#     lmList, bboxInfo = detector.findPosition(pose_img, bboxWithHands=False, draw=False)
    
#     shirts = get_shirt_list()
#     if not shirts:
#         raise ValueError("No shirt images found")

#     if lmList and len(lmList) > 24:
#         shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
#         imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
#         if imgShirt is None or imgShirt.shape[2] != 4:
#             raise ValueError(f"Failed to load valid shirt image with alpha channel: {shirt_path}")

#         # Extract keypoints - using more reliable landmarks
#         left_shoulder = np.array(lmList[11][1:3])
#         right_shoulder = np.array(lmList[12][1:3])
#         # Use waist instead of hips for better fit
#         left_waist = np.array(lmList[23][1:3])
#         right_waist = np.array(lmList[24][1:3])

#         # Calculate midpoints
#         mid_shoulder = (left_shoulder + right_shoulder) / 2
#         mid_waist = (left_waist + right_waist) / 2
        
#         # Calculate torso dimensions
#         torso_width = np.linalg.norm(left_shoulder - right_shoulder)
#         torso_height = np.linalg.norm(mid_shoulder - mid_waist)
        
#         # Add proportional padding
#         width_padding = int(torso_width * 0.3)
#         height_padding_top = int(torso_height * 0.1)
#         height_padding_bottom = int(torso_height * 0.2)
        
#         # Calculate shirt position (adjusted proportions)
#         shirt_width = int(torso_width + width_padding)
#         shirt_height = int(torso_height + height_padding_top + height_padding_bottom)
        
#         # Calculate shirt position
#         shirt_top = int(mid_shoulder[1] - height_padding_top)
#         shirt_bottom = int(mid_waist[1] + height_padding_bottom)
#         shirt_left = int(mid_shoulder[0] - shirt_width//2)
#         shirt_right = int(mid_shoulder[0] + shirt_width//2)

#         # Form destination points (perspective transform)
#         dst_pts = np.float32([
#             [shirt_left, shirt_top],          # top-left
#             [shirt_right, shirt_top],         # top-right
#             [shirt_right, shirt_bottom],      # bottom-right
#             [shirt_left, shirt_bottom]        # bottom-left
#         ])
        
#         # Source points (shirt corners)
#         src_pts = np.float32([
#             [0, 0],
#             [imgShirt.shape[1], 0],
#             [imgShirt.shape[1], imgShirt.shape[0]],
#             [0, imgShirt.shape[0]]
#         ])

#         # Calculate perspective transform
#         matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
#         # Warp shirt with better interpolation
#         warped = cv2.warpPerspective(
#             imgShirt, 
#             matrix, 
#             (img.shape[1], img.shape[0]),
#             flags=cv2.INTER_LANCZOS4,
#             borderMode=cv2.BORDER_CONSTANT,
#             borderValue=(0, 0, 0, 0)
#         )

#         # Enhanced overlay with blending
#         result = overlay_transparent(img, warped, alpha_blend=0.95)
        
#         # Save processed image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         processed_filename = f"processed_{timestamp}.jpg"
#         processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
#         cv2.imwrite(processed_path, result)

#         return processed_filename
#     else:
#         raise ValueError("Pose detection failed - could not find body landmarks")

# def process_image(user_image_path, shirt_index):
#     detector = PoseDetector()
#     img = cv2.imread(user_image_path)
#     if img is None:
#         raise ValueError("Failed to load user image")

#     # Create a copy for pose detection
#     img_pose = img.copy()
#     img_pose = detector.findPose(img_pose)
#     lmList, bboxInfo = detector.findPosition(img_pose, bboxWithHands=False, draw=False)
    
#     shirts = get_shirt_list()
#     if not shirts:
#         raise ValueError("No shirt images found")

#     if lmList and len(lmList) > 24:
#         shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirts[shirt_index])
#         imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
#         if imgShirt is None or imgShirt.shape[2] != 4:
#             raise ValueError(f"Failed to load valid shirt image with alpha channel: {shirt_path}")

#         # Get key points (using more reliable landmarks)
#         left_shoulder = np.array(lmList[11][1:3])
#         right_shoulder = np.array(lmList[12][1:3])
#         left_hip = np.array(lmList[23][1:3])
#         right_hip = np.array(lmList[24][1:3])

#         # Calculate torso center
#         torso_center_x = int((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4)
#         torso_center_y = int((left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4)

#         # Calculate torso dimensions
#         torso_width = int(np.linalg.norm(left_shoulder - right_shoulder) * 1.8)
#         torso_height = int(np.linalg.norm(left_shoulder - left_hip) * 1.5)

#         # Calculate shirt position
#         shirt_top = torso_center_y - int(torso_height * 0.4)
#         shirt_bottom = torso_center_y + int(torso_height * 0.6)
#         shirt_left = torso_center_x - int(torso_width / 2)
#         shirt_right = torso_center_x + int(torso_width / 2)

#         # Ensure coordinates are within image bounds
#         h, w = img.shape[:2]
#         shirt_top = max(0, shirt_top)
#         shirt_bottom = min(h, shirt_bottom)
#         shirt_left = max(0, shirt_left)
#         shirt_right = min(w, shirt_right)

#         # Calculate scale factors
#         scale_x = (shirt_right - shirt_left) / imgShirt.shape[1]
#         scale_y = (shirt_bottom - shirt_top) / imgShirt.shape[0]
        
#         # Resize shirt while maintaining aspect ratio
#         scale = min(scale_x, scale_y) * 0.95  # Slightly undersize to fit better
#         new_width = int(imgShirt.shape[1] * scale)
#         new_height = int(imgShirt.shape[0] * scale)
#         resized_shirt = cv2.resize(imgShirt, (new_width, new_height))

#         # Calculate position to center the shirt
#         x_offset = torso_center_x - new_width // 2
#         y_offset = shirt_top
        
#         # Create overlay mask
#         overlay = img.copy()
#         alpha_s = resized_shirt[:, :, 3] / 255.0
#         alpha_l = 1.0 - alpha_s

#         # Blend the shirt onto the overlay
#         for c in range(0, 3):
#             try:
#                 overlay[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = (
#                     alpha_s * resized_shirt[:, :, c] + 
#                     alpha_l * overlay[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c]
#                 )
#             except:
#                 # Handle boundary issues
#                 pass

#         # Final blend with original image
#         alpha = 0.7  # Blend factor
#         result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

#         # Save processed image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         processed_filename = f"processed_{timestamp}.jpg"
#         processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
#         cv2.imwrite(processed_path, result)

#         return processed_filename
#     else:
#         raise ValueError("Pose detection failed - could not find body landmarks")



def process_image(user_image_path, shirt_index):
    detector = PoseDetector()
    img = cv2.imread(user_image_path)
    if img is None:
        raise ValueError("Failed to load user image")

    # Create pose visualization
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

        # Get original landmarks
        left_shoulder = np.array(lmList[11][1:3])
        right_shoulder = np.array(lmList[12][1:3])
        left_hip = np.array(lmList[23][1:3])
        right_hip = np.array(lmList[24][1:3])

        # Calculate center point
        center_x = np.mean([left_shoulder[0], right_shoulder[0], 
                            left_hip[0], right_hip[0]])
        center_y = np.mean([left_shoulder[1], right_shoulder[1], 
                            left_hip[1], right_hip[1]])
        
        # Calculate dimensions with scale factor
        scale = 1.5
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scale
        hip_height = abs(left_hip[1] - left_shoulder[1]) * scale

        # Create synthetic torso points
        left_shoulder = [center_x - shoulder_width / 2, center_y - hip_height / 2]
        right_shoulder = [center_x + shoulder_width / 2, center_y - hip_height / 2]
        left_hip = [center_x - shoulder_width / 2, center_y + hip_height / 2]
        right_hip = [center_x + shoulder_width / 2, center_y + hip_height / 2]

        # Perspective transformation
        src_pts = np.float32([
            [0, 0],  # top-left
            [imgShirt.shape[1], 0],  # top-right
            [imgShirt.shape[1], imgShirt.shape[0]],  # bottom-right
            [0, imgShirt.shape[0]]   # bottom-left
        ])
        
        # Apply vertical offset to shoulders (+30px)
        dst_pts = np.float32([
            [left_shoulder[0], left_shoulder[1] + 30],  # top-left
            [right_shoulder[0], right_shoulder[1] + 30],  # top-right
            [right_hip[0], right_hip[1]],  # bottom-right
            [left_hip[0], left_hip[1]]   # bottom-left
        ])

        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0, 0))
        
        # Warp shirt with border handling
        # warped = cv2.warpPerspective(
        #     imgShirt, 
        #     matrix, 
        #     (img.shape[1], img.shape[0]),
        #     flags=cv2.INTER_LANCZOS4,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=(0, 0, 0, 0)
        # )

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
