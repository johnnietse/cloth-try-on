# Virtual Clothing Try-On Web App

This is a web app that allows users to upload an image of their chosen clothing (with the background removed) and a video of themselves (one person, facing the camera) to virtually try on the clothing. After uploading the necessary files and selecting the desired clothing, users can generate a video where the selected clothing is applied to their image.

### Key Features:
- **Upload Clothing Image**: Users can upload an image of clothing with the background removed (e.g., PNG format).
- **Upload Personal Video**: Users upload a video of themselves facing the camera.
- **Virtual Try-On**: The app combines the uploaded video and clothing image to generate a video of the clothing virtually fitting the user.
- **Download Processed Video**: After processing, users are notified when the video is ready for download, with a link to download the final video.

## Prerequisites

To run this project locally, you will need to set up a virtual environment and install the required dependencies. Here's how to do it:

**Before cloning the repository, change your working directory to the folder where you want the project to be saved:**

Navigate to the directory where you want to store the project:
```bash
cd /path/to/your/directory
```

### 1. Clone the repository:
```bash
git clone https://github.com/johnnietse/cloth-virtual-try-on.git
cd cloth-virtual-try-on
```

### 2. Set up a virtual environment:
If you're using Python 3, you can set up a virtual environment with the following commands:
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment:
- For Windows:
```bash
venv\Scripts\activate
```

- For macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Install dependencies:
Once your virtual environment is activated, install the required dependencies using **requirements.txt**:
```bash
pip install -r requirements.txt
```

## Running the App Locally
### 1. Start the Flask server:
After setting up the environment and installing dependencies, you can run the web app using the following command:
```bash
flask run
```

This will start a local development server, and you should see an output like this in the terminal:
```bash
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### 2. Access the Web App:
Once the server is running, open your web browser and go to the following URL:
```bash
http://127.0.0.1:5000
```
You should see the web app interface where you can upload your clothing image and personal video.

### 3. Upload Files and Process:
- Upload your **clothing image** (make sure the background is removed).
- Upload your **personal video** (where you are facing the camera).
- Select the clothing item you want to try on.
- Click the **"Upload and Process"** button to generate the virtual try-on video.
  
### 4. Download the Processed Video:
Once the video has been processed, you will see a message:

```bash
Video processing complete! Click the link below to download.
```

Click on the **"Download Processed Video"** button to download the generated video.

## Sample Files
To get an idea of how the app works, you can look at the following sample files within the repository:

- Clothing Images: Sample clothing images can be found in the **Resources/Shirts** folder.
- Uploaded Videos: Sample user videos can be found in the **static/uploads** folder.
- Processed Videos: Sample processed virtual clothing try-on videos are available in the **static/processed** folder.

These samples demonstrate the input and output for the app's functionality.

## Future Considerations
- Deployment to a cloud platform (e.g., Render, Heroku, AWS).
- Enhancements to virtual clothing fitting (e.g., improved image processing and fit accuracy).
- Support for multiple clothing items or different video orientations.

---

## 📸 Screenshot
![virtual-try-on](https://github.com/user-attachments/assets/abc117df-a0ee-4a15-ac30-6964c31e48e4)

---

## Technical Overview (Updated)

### Backend Framework

- **Flask**: Powers the web app with routes for uploading videos/shirts, processing, and downloads. Uses <code>render_template</code> for dynamic HTML rendering and <code>send_from_directory</code> for file delivery.

- **Dynamic Shirt Management**: Shirt images are stored in <code>Resources/Shirts</code> and fetched dynamically via <code>get_shirt_list()</code> for real-time updates.

### Pose Detection & Advanced Image Processing
- **Pose Detection**: Leverages <code>cvzone.PoseModule</code> to detect body landmarks (shoulders, hips) for precise clothing placement.

- **Perspective Transformation**: Uses <code>cv2.getPerspectiveTransform</code> and <code>cv2.warpPerspective</code> to warp clothing images onto the user’s body based on detected landmarks, ensuring realistic alignment.

- **Green Screen Removal**: The <code>overlay_transparent</code> function removes green backgrounds from clothing images while applying semi-transparency for natural blending.

### File Handling & Storage
- **Filesystem Storage**: Uploaded videos and processed outputs are stored in <code>static/uploads</code> and <code>static/processed</code>, respectively. Filenames include timestamps to avoid collisions.

- **Direct File Serving**: Processed videos are served directly from the filesystem using <code>send_from_directory</code>, eliminating the need for a database.

### Video Processing Pipeline
1. **Frame Capture**: Video frames are extracted using <code>cv2.VideoCapture</code>.

2. **Landmark Detection**: Shoulder/hip landmarks are identified to define target regions for clothing placement.

3. **Dynamic Clothing Adjustment**:

    - Bounding Box Scaling: Expands the clothing region using a scaling factor for better coverage.
    
    - Perspective Warping: Warps the clothing image to match the user’s pose using a computed transformation matrix.

4. **Transparency Blending**: Overlays the warped clothing onto each frame with adjustable opacity and green-screen removal.

5. **Video Reconstruction**: Compiled into an MP4 file using <code>cv2.VideoWriter</code>.

### Key Features

- **Dynamic Shirt Uploads**: Users can upload new clothing images (PNG/JPG) via <code>/upload_shirt</code>, which are immediately available for try-on.

- **Error Handling**: Basic checks for file validity and pose detection failures, with JSON error responses for API routes.

- **Environment Configuration**: Uses <code>python-dotenv</code> for environment variables (if needed), though PostgreSQL integration is currently inactive.

### Dependencies
- **Core Libraries**:
    
    - <code>Flask</code>: Web framework.
    
    - <code>OpenCV (cv2)</code>: Video/image processing.
    
    - <code>cvzone</code>: Pose detection utilities.
    
    - <code>python-dotenv</code>: Environment variable management (optional).

- **Runtime**: Requires <code>psycopg2</code> (though not actively used here) and <code>numpy</code> for array operations.

### Deployment Notes
- **Development Mode**: Runs with <code>debug=True</code> for easy testing (not suitable for production).

- **Scalability**: Designed for filesystem storage, making it deployable to platforms like Heroku or Render with ephemeral storage. For production, consider cloud storage (AWS S3) and a task queue (Celery) for video processing.

---

This revised technical overview reflects the current codebase’s focus on filesystem-based storage, advanced perspective warping, and dynamic clothing management. 

---
## 🎥 Demo Video
A short walkthrough of the application is available below.

▶️ Watch the full demo here:
https://youtu.be/Xz_QCwU1M5I
