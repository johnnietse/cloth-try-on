// Wait for the document to load
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const statusMessage = document.getElementById('statusMessage');
    const videoWrapper = document.getElementById('videoWrapper');
    const processedVideo = document.getElementById('processedVideo');
    const processButton = document.getElementById('processButton');
    let uploadedVideoPath = '';

    // Event listener for the form submission (video upload)
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        statusMessage.innerHTML = 'Uploading video...';

        const formData = new FormData();
        const videoFile = document.getElementById('video').files[0];
        formData.append('video', videoFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const text = await response.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch {
                throw new Error('Server response is not valid JSON: ' + text);
            }

            if (!response.ok || data.status === "error") {
                throw new Error(data?.error || "Upload failed");
            }

            uploadedVideoPath = data.filePath;
            statusMessage.innerHTML = 'Video uploaded successfully. Click the button to process the video.';
            processButton.style.display = 'inline-block';
        } catch (error) {
            console.error('Error during video upload:', error);
            statusMessage.innerHTML = 'An error occurred while uploading the video. ' + error.message;
        }
    });

    // Function to handle the processing of the uploaded video
    window.processVideo = async () => {
        statusMessage.innerHTML = 'Processing video...';

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ videoPath: uploadedVideoPath })
            });

            const text = await response.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch {
                throw new Error('Server response is not valid JSON: ' + text);
            }

            if (!response.ok || data.status === "error") {
                throw new Error(data?.error || "Processing failed");
            }

            const outputPath = data.outputPath;
            statusMessage.innerHTML = 'Video processed successfully. Loading the processed video...';
            processedVideo.src = outputPath;
            videoWrapper.style.display = 'block';
            statusMessage.innerHTML = 'Processing complete. Enjoy the video!';
        } catch (error) {
            console.error('Error during video processing:', error);
            statusMessage.innerHTML = 'An error occurred while processing the video. ' + error.message;
        }
    };
});
