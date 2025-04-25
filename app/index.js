const video = document.getElementById("video");
const resultDiv = document.getElementById("result");
let mediaRecorder, chunks = [], recording = false;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if(e.data.size) chunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStop;
}

async function onRecordingStop() {
  const blob = new Blob(chunks, { type: "video/webm" });
  chunks = [];

  resultDiv.innerText = "Enviando video al servidor…";
  const form = new FormData();
  form.append("file", blob, "sign.webm");

  try {
    const resp = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: form
    });
    const data = await resp.json();
    resultDiv.innerText = `Predicción: ${data.sign} (confianza ${(data.confidence*100).toFixed(1)}%)`;
  } catch (err) {
    console.error(err);
    resultDiv.innerText = "❌ Error al predecir";
  }
}

function toggleRecording() {
  if (!mediaRecorder) return;
  if (recording) mediaRecorder.stop();
  else { chunks = []; mediaRecorder.start(); }
  recording = !recording;
}

window.addEventListener("keydown", e => {
  if (e.code === "Space") toggleRecording();
});

startCamera();
