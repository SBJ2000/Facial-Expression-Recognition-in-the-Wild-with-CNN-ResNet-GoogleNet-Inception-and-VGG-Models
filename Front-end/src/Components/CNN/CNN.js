import React, { useEffect, useRef, useState } from "react";
import { Container, Row, Col, Button, Form } from "react-bootstrap";
import Particle from "../Particle";

function CNN() {
  const videoRef = useRef();
  const canvasRef = useRef();
  const fileInputRef = useRef();
  const [isCameraPaused, setIsCameraPaused] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const videoElement = videoRef.current;
        if (videoElement) {
          videoElement.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };

    initCamera();
  }, []);

  const handlePredictClick = () => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;

    if (videoElement && canvasElement) {
      // Draw the current frame from the video onto the canvas
      const context = canvasElement.getContext("2d");
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

      // Pause the video to freeze the current frame
      videoElement.pause();
      setIsCameraPaused(true);

      // Get the image data from the canvas
      const imageData = canvasElement.toDataURL("image/png");

      // Send the image data to the backend (you need to implement this part)
      sendImageToBackend(imageData);
    }
  };

  const handleFileUpload = () => {
    const fileInput = fileInputRef.current;

    if (fileInput && fileInput.files.length > 0) {
      const file = fileInput.files[0];

      // Read the file as Data URL
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageData = event.target.result;

        // Set the selected image state to display on the frontend
        setSelectedImage(imageData);

        // Send the image data to the backend (you need to implement this part)
        sendImageToBackend(imageData);
      };

      reader.readAsDataURL(file);
    }
  };

  const handleRetakeClick = () => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const predictionDiv = document.getElementById('predictionDiv');

    if (videoElement && canvasElement && predictionDiv) {
      // Resume the video stream
      videoElement.play();
      setIsCameraPaused(false);
      predictionDiv.innerHTML = '';
    }
  };

  const sendImageToBackend = (imageData) => {
    // Create a FormData object to send the image file
    const formData = new FormData();
    formData.append('image', imageData);
  
    // Make a POST request to the Flask endpoint
    fetch('http://127.0.0.1:5000/CNN', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.text())
      .then(message => {
        // Handle the response from the server
        console.log('Server response:', message);
        
        const emotionMapping = {
          0: ' 😡 Angry',
          1: ' 😷 Disgust',
          2: ' 😨 Fear',
          3: ' 😊 Happy',
          4: ' 😐 Neutral',
          5: ' 😢 Sad',
          6: ' 😲 Surprise',
        };

        const predictionDiv = document.getElementById('predictionDiv');
        if (predictionDiv) {
          // Assume 'message' contains the numerical prediction
          const jsonResponse = JSON.parse(message);
  
          if (jsonResponse.success) {
            const numericalPrediction = jsonResponse.prediction;
            const predictedEmotion = emotionMapping[numericalPrediction];
  
            // Display the predicted emotion inside the predictionDiv
            predictionDiv.innerHTML += `<p style="color: #fff; text-align: center; font-size: 18px;">Prediction: <strong> <br>${predictedEmotion}</strong></p>`;
          } else {
            // Handle the case when success is false
            console.error('Prediction failed:', jsonResponse.error);
          }
        }
      })
      .catch(error => {
        // Handle any errors that occurred during the fetch
        console.error('Error sending image data:', error);
      });
  };

  return (
    <Container fluid className="project-section">
      <Container>
        {/* ... (previous code) */}
        <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
          <Col md={8} className="project-card">
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <video
                ref={videoRef}
                width="75%"
                height="auto"
                autoPlay
                playsInline
                muted
                id="cameraFeed"
                style={{ border: "2px solid #fff", borderRadius: "8px" }}
              />
              <div
                style={{
                  width: "20%",
                  backgroundColor: "#333",
                  padding: "10px",
                  borderRadius: "8px",
                }}
              >
                <p style={{ color: "#fff", textAlign: "center" }}>Prediction</p>
                <div id="predictionDiv"></div>
                {selectedImage && (
                  <div style={{ marginTop: "10px" }}>
                    <p style={{ color: "#fff", textAlign: "center" }}>Selected Image</p>
                    <img
                      src={selectedImage}
                      alt="Selected"
                      style={{ width: "100%", borderRadius: "8px" }}
                    />
                  </div>
                )}
                {isCameraPaused ? (
                  <Button
                    variant="success"
                    onClick={handleRetakeClick}
                    style={{ marginTop: "10px" }}
                  >
                    Retake
                  </Button>
                ) : (
                  <>
                    <Form.Group controlId="fileUpload" style={{ marginTop: "10px" }}>
                      <Form.Label style={{ color: "#fff" }}>Upload Image</Form.Label>
                      <Form.Control type="file" ref={fileInputRef} accept="image/*" />
                    </Form.Group>
                    <Button
                      variant="primary"
                      onClick={handleFileUpload}
                      style={{ marginTop: "10px" }}
                    >
                      Upload
                    </Button>
                    <Button
                      variant="primary"
                      onClick={handlePredictClick}
                      style={{ marginTop: "10px" }}
                    >
                      Predict
                    </Button>
                  </>
                )}
              </div>
            </div>
          </Col>
        </Row>
        <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
      </Container>
    </Container>
  );
}

export default CNN;