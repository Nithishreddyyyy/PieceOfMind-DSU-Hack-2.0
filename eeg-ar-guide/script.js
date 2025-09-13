const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  videoElement.srcObject = stream;
});

const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});

camera.start();

function getWireColorNear(x, y, offsetY = -20, size = 10) {
  const imageData = canvasCtx.getImageData(x - size / 2, y + offsetY - size / 2, size, size);
  const data = imageData.data;
  let r = 0, g = 0, b = 0;

  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }

  const pixelCount = data.length / 4;
  return {
    r: Math.round(r / pixelCount),
    g: Math.round(g / pixelCount),
    b: Math.round(b / pixelCount)
  };
}

function detectWireColor({ r, g, b }) {
    const brightness = (r + g + b) / 3;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const diff = max - min;
    // White: all channels high, brightness very high
    if (brightness > 130 && diff < 30) return 'White';
  
    // Black: all channels low, brightness low
    if (r < 70 && g < 70 && b < 70 && brightness < 80) return 'Black';
  
    // Red: red dominant, green and blue suppressed
    if (r > 130 && g < 100 && b < 100) return 'Red';
  
    // Cyan / Bluish Green: blue and green both strong
    if (g > 100 && b > 100 && r < 100) return 'BluishGreen';
  
    // Yellow: red and green strong, blue low
    if (r > 180 && g > 180 && b < 120) return 'Yellow';
  
    return 'None';
  }
  

  function drawWireFeedback(x, y, expectedLabel, expectedColor) {
    const colorSample = getWireColorNear(x, y);
    const detected = detectWireColor(colorSample);
    const isCorrect = detected === expectedColor;
  
    canvasCtx.beginPath();
    canvasCtx.arc(x, y, 7, 0, 2 * Math.PI);
    canvasCtx.fillStyle = isCorrect ? 'green' : 'red';
    canvasCtx.fill();
  
    canvasCtx.font = "12px Arial";
    canvasCtx.fillStyle = isCorrect ? 'green' : 'red';
    canvasCtx.fillText(`${expectedLabel}`, x + 8, y);
  
    return isCorrect;
  }
  let videoStream = null;

navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  videoElement.srcObject = stream;
  videoStream = stream;
});

document.getElementById("stopButton").addEventListener("click", () => {
  // Stop the camera
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
  }

  // Stop Mediapipe faceMesh
  camera.stop(); // Stop the camera processing frames
  faceMesh.close(); // Free up Mediapipe resources

  // Clear canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Optionally, hide video/canvas
  videoElement.style.display = "none";
  canvasElement.style.display = "none";

  // Disable the button after use
  document.getElementById("stopButton").disabled = true;
  document.getElementById("stopButton").innerText = "Started";
});

  function onResults(results) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
    if (results.multiFaceLandmarks.length > 0) {
      const landmarks = results.multiFaceLandmarks[0];
  
      const fp1 = landmarks[299];  // left forehead
      const fp2 = landmarks[69];   // right forehead
      const ref = landmarks[132];  // reference point (ear)
  
      const x1 = fp1.x * canvasElement.width;
      const y1 = fp1.y * canvasElement.height;
      const x2 = fp2.x * canvasElement.width;
      const y2 = fp2.y * canvasElement.height;
      const xr = ref.x * canvasElement.width;
      const yr = ref.y * canvasElement.height;
  
      drawWireFeedback(x1, y1, "+ve(white)", "White");
      drawWireFeedback(x2, y2, "-ve(black)", "Black");
      drawWireFeedback(xr, yr, "Ref(blue)", "BluishGreen");
    }
  
    canvasCtx.restore();
  }
  
