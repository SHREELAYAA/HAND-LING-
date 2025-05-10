const videoElement = document.getElementById('videoElement');
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing webcam: ', err);
    });

function captureImage() {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/png');

    fetch('/recognize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `video_data=${encodeURIComponent(dataURL)}`
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').innerText = `Predicted Gesture: ${data.recognized_sign}`;
        speakPrediction(data.recognized_sign);
    })
    .catch(err => {
        console.error('Error predicting gesture: ', err);
    });
}

function speakPrediction(prediction) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(prediction);
    utterance.volume=1;
    utterance.rate=1;
    utterance.pitch=1;
    synth.speak(utterance);
}
