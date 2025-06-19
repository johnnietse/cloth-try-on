# Virtual Clothing Try-On Web App (Image Version)

This is a web app that allows users to upload an image of their chosen clothing (with the background removed) and an image of themselves (one person, facing the camera) to virtually try on the clothing. After uploading the necessary files and selecting the desired clothing, users can generate an image where the selected clothing is applied to their image.

### Key Features:
- **Upload Clothing Image**: Users can upload an image of clothing with the background removed (e.g., PNG format).
- **Upload Personal Image**: Users upload an image of themselves facing the camera.
- **Virtual Try-On**: The app combines the uploaded image and clothing image to generate a image of the clothing virtually fitting the user.
- **Download Processed Image**: After processing, users are notified when the image is ready for download, with a link to download the final image.

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
You should see the web app interface where you can upload your clothing image and personal image.

### 3. Upload Files and Process:
- Upload your **clothing image** (make sure the background is removed).
- Upload your **personal image** (where you are facing the camera).
- Select the clothing item you want to try on.
- Click the **"Upload and Process"** button to generate the virtual try-on image.
  
### 4. Download the Processed Image:
Once the image has been processed, you will see a button where it will ask you to download the image:

```bash
Click on the **"Download Processed Image"** button to download the generated image.
```

## Sample Files
To get an idea of how the app works, you can look at the following sample files within the repository:

- Clothing Images: Sample clothing images can be found in the `static/shirts` folder.

These samples demonstrate the input for the app's functionality.

## Future Considerations
- (Completed) Deployment to a cloud platform (e.g., Render, Heroku, AWS).
- Enhancements to virtual clothing fitting (e.g., improved image processing and fit accuracy).
- Support for multiple clothing items or different image orientations.

---

## Technical Overview (Updated)

### Backend Framework

- **Flask**: Powers the web app with routes for uploading images/shirts, processing, and downloads. Uses <code>render_template</code> for dynamic HTML rendering and <code>send_from_directory</code> for file delivery.

- **Dynamic Shirt Management**: Shirt images are stored in <code>static/shirts</code> and fetched dynamically via <code>get_shirt_list()</code> for real-time updates.

### Pose Detection & Advanced Image Processing
- **Pose Detection**: Leverages <code>cvzone.PoseModule</code> to detect body landmarks (shoulders, hips) for precise clothing placement.

- **Perspective Transformation**: Uses <code>cv2.getPerspectiveTransform</code> and <code>cv2.warpPerspective</code> to warp clothing images onto the user’s body based on detected landmarks, ensuring realistic alignment.

- **Green Screen Removal**: The <code>overlay_transparent</code> function removes green backgrounds from clothing images while applying semi-transparency for natural blending.

### File Handling & Storage
- **Filesystem Storage**: Uploaded images and processed outputs are stored in <code>static/uploads</code> and <code>static/processed</code>, respectively. 

- **Direct File Serving**: Processed images are served directly from the filesystem using <code>send_from_directory</code>, eliminating the need for a database. But to show you the sample clothing images, I still connect a database on Render for the deployment.

### Image Processing Pipeline

1. **Landmark Detection**: Shoulder/hip landmarks are identified to define target regions for clothing placement.

2. **Dynamic Clothing Adjustment**:

    - Bounding Box Scaling: Expands the clothing region using a scaling factor for better coverage.
    
    - Perspective Warping: Warps the clothing image to match the user’s pose using a computed transformation matrix.

3. **Transparency Blending**: Overlays the warped clothing onto each frame with adjustable opacity and green-screen removal.

### Key Features

- **Dynamic Shirt Uploads**: Users can upload new clothing images (PNG/JPG) via <code>/upload_shirt</code>, which are immediately available for try-on.

- **Error Handling**: Basic checks for file validity and pose detection failures, with JSON error responses for API routes.

- **Environment Configuration**: Uses <code>python-dotenv</code> for environment variables (if needed), including PostgreSQL integration.

### Dependencies
- **Core Libraries**:
    
    - <code>Flask</code>: Web framework.
    
    - <code>OpenCV (cv2)</code>: Image processing.
    
    - <code>cvzone</code>: Pose detection utilities.
    
    - <code>python-dotenv</code>: Environment variable management (optional).

- **Runtime**: Requires <code>psycopg2</code> (though not actively used here) and <code>numpy</code> for array operations.

### Deployment Notes
- **Development Mode**: Runs with <code>debug=True</code> for easy testing (not suitable for production).

- **Scalability**: Designed for filesystem storage, making it deployable to platforms like Heroku or Render with ephemeral storage. 

---

This revised technical overview reflects the current codebase’s focus on filesystem-based storage, advanced perspective warping, and dynamic clothing management. 

---

## Virtual Clothing Try-On Web App Deployment Guide

This guide walks you through setting up the PostgreSQL database and deploying the Flask web application on Render.

### Prerequisites
- Render account
- GitHub account with repository access
- Credit card (for database persistence beyond free tier)

### 1. Database Setup on Render
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → Select "PostgreSQL"
3. Configure database:
   - **Name**: `cloth-try-on-db` (could be rename to anything)
   - **Region**: `Virginia (US East)` (could be anything)
   - **PostgreSQL Version**: `16`
   - **Instance Type**: `Free` (upgrade later if needed)
4. Click "Create Database"
5. Wait for database to become `available` (takes 2-3 minutes)
6. Note these connection details from the "Connect" tab:

### 2. Web Service Deployment
1. In Render Dashboard, click "New +" → Select "Web Service"
2. Connect your GitHub repository:
- Select `johnnietse/cloth-try-on` (this will vary depending on how you name your github repository)
- Choose branch: `main` (this will vary too depending on how you name your branch, but usually the branch will come will the naming of "main")

### 3. Configure service:
- **Name**: `virtual-try-on` (could be rename to anything)
- **Region**: `Virginia (US East)` (could be anything)
- **Instance Type**: `Free` (upgrade later if needed)

### 4. Set build commands:
- Build Command:
```bash
pip install --upgrade pip && 
pip install -r requirements.txt && 
chmod +x preinstall.sh && 
./preinstall.sh
```
- Start Command:
```bash
gunicorn --config gunicorn_config.py app:app
```

### 5. Add environment variables under "Environment" tab:
| Key                | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| DATABASE_URL       | `postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}`  |
| DB_HOST            | `[Your database hostname]`                                            |
| DB_NAME            | `[Your database name]`                                                |
| DB_PASSWORD        | `[Your database password]`                                            |
| DB_PORT            | `[Your database port #]`                                              |
| DB_USER            | `[Your database username]`                                            |
| FLASK_APP          | `app.py`                                                              |
| FLASK_DEBUG        | `0`                                                                   | 
| FLASK_ENV          | `production`                                                          |
| PROCESSED_FOLDER   | `[Your processed folder path]`                                        |
| PYTHON_VERSION     | `3.10.12`                                                             |
| SHIRT_FOLDER       | `[Your shirt folder path]`                                            |
| UPLOAD_FOLDER      | `[Your upload folder path]`                                           |

Note that since it is an internal deployment on Render (not an external deployment) as the POSTGRESQL database can also be used externally too, I fill in the Internal Database URL as the value of the `DATABASE_URL` key instead of the External Database URL.

### 6. Click "Create Web Service"

### 7. Post-Deployment
First build will take 5-10 minutes (installs dependencies and runs preinstall.sh). Health check endpoint: /healthz


### 8. Maintenance
- Database Backups: Free tier doesn't include automatic backups. Upgrade for backup functionality
- Web Service:
  - Free instances sleep after 15 minutes inactivity
  - Wake-up takes 30-60 seconds
- To update deployment:
  1. Push changes to main branch
  2. Render will auto-deploy updates

### 9. Troubleshooting
1. Check build logs in Render dashboard
2. Verify all environment variables are set correctly
3. Ensure database is in "available" state
4. Check free tier resource limits (storage, RAM, CPU)


### Key Notes:
1. **Database Security**: 
   - Your current `0.0.0.0/0` access rule allows global connections
   - For production: Restrict to Render's IP ranges or use private networking

2. **Environment Variables**:
   - Replace bracketed values `[ ]` with actual credentials/paths
   - The `DATABASE_URL` should follow format: `postgresql://USER:PASSWORD@HOST:PORT/DATABASE`

3. **Free Tier Limitations**:
   - Database: Auto-deletes after 90 days
   - Web Service: Sleeps when inactive
   - Storage: Limited to 1GB (monitor usage in dashboard)

4. **Preinstall Script**:
   - Ensure `preinstall.sh` has necessary permissions in your repo
   - Include any model downloads/asset preparations here

---

## 🎥 Demo Video
A short walkthrough of the application is available below.

▶️ Watch the full demo here:
https://youtu.be/Xz_QCwU1M5I


---

## 📸 Screenshots
![Screenshot (3615)](https://github.com/user-attachments/assets/697d0201-84e0-41e2-884b-6249c9a28338)
![Screenshot (3614)](https://github.com/user-attachments/assets/9c4635e9-cb8b-4fe1-ad7d-99b5911be6c1)

