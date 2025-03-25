const video = document.getElementById("video");
let mediaRecorder;
let chunks = [];
let recording = false;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: "video/webm" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "recorded-video.webm";
            a.click();
            chunks = [];
        };
    } catch (error) {
        console.error("Error al acceder a la cámara:", error);
    }
}

function toggleRecording() {
    if (!mediaRecorder) return;
    if (recording) {
        mediaRecorder.stop();
    } else {
        chunks = [];
        mediaRecorder.start();
    }
    recording = !recording;
}

window.addEventListener("keydown", event => {
    console.log('a')
    if (event.code === "Space") {
        toggleRecording();
    }
});

startCamera();