// Wait for the document to load
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const statusMessage = document.getElementById('statusMessage');
    const videoWrapper = document.getElementById('videoWrapper');
    const processedVideo = document.getElementById('processedVideo');

    // Event listener for the form submission (video upload)
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        // Clear any previous status messages
        statusMessage.innerHTML = 'Uploading video...';

        // Create form data to send the video file
        const formData = new FormData();
        const videoFile = document.getElementById('video').files[0];
        formData.append('video', videoFile);

        try {
            // Upload the video to the server
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await uploadResponse.json();
            const videoPath = data.filePath;

            // Notify user about the video upload
            statusMessage.innerHTML = 'Video uploaded successfully. Now processing the video...';

            // Process the uploaded video
            const processResponse = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ videoPath })
            });

            const processData = await processResponse.json();
            const outputPath = processData.outputPath;

            // Notify user when processing is complete
            statusMessage.innerHTML = 'Video processed successfully. Loading the processed video...';

            // Set the processed video URL and display it
            processedVideo.src = outputPath;
            videoWrapper.style.display = 'block';
            statusMessage.innerHTML = 'Processing complete. Enjoy the video!';
        } catch (error) {
            // Handle errors
            console.error('Error during video processing:', error);
            statusMessage.innerHTML = 'An error occurred while processing the video. Please try again.';
        }
    });
});


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

        // Clear any previous status messages
        statusMessage.innerHTML = 'Uploading video...';

        // Create form data to send the video file
        const formData = new FormData();
        const videoFile = document.getElementById('video').files[0];
        formData.append('video', videoFile);

        try {
            // Upload the video to the server
//            const uploadResponse = await fetch('/upload', {
//                method: 'POST',
//                body: formData
//            });
//
//            const data = await uploadResponse.json();

            const response = await fetch('/upload', {
              method: 'POST',
              body: formData,
              signal: AbortSignal.timeout(250000)
            });

            if (!response.ok) {
              const error = await response.json().catch(() => ({ error: "Unknown error" }));
              throw new Error(error.message || "Upload failed");
            }

            const data = await response.json();
            console.log("Backend response:", data);  // Debugging
            if (data.status === "error") throw new Error(data.error);

            uploadedVideoPath = data.filePath;

            // Notify user about the video upload
            statusMessage.innerHTML = 'Video uploaded successfully. Click the button to process the video.';
            processButton.style.display = 'inline-block';  // Show the process button
        } catch (error) {
            // Handle errors
            console.error('Error during video upload:', error);
            statusMessage.innerHTML = 'An error occurred while uploading the video. Please try again.';
        }
    });

    // Function to handle the processing of the uploaded video
    window.processVideo = async () => {
        // Notify user that processing is starting
        statusMessage.innerHTML = 'Processing video...';

        try {
            // Process the uploaded video
            const processResponse = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ videoPath: uploadedVideoPath })
            });

            const processData = await processResponse.json();
            const outputPath = processData.outputPath;

            // Notify user when processing is complete
            statusMessage.innerHTML = 'Video processed successfully. Loading the processed video...';

            // Set the processed video URL and display it
            processedVideo.src = outputPath;
            videoWrapper.style.display = 'block';
            statusMessage.innerHTML = 'Processing complete. Enjoy the video!';
        } catch (error) {
            // Handle errors
            console.error('Error during video processing:', error);
            statusMessage.innerHTML = 'An error occurred while processing the video. Please try again.';
        }
    };
});





