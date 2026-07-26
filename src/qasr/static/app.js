const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
const languageSelect = document.querySelector("#language");
const transcript = document.querySelector("#transcript");
const placeholder = document.querySelector("#placeholder");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const timing = document.querySelector("#timing");
const levelBar = document.querySelector("#level-bar");
const timer = document.querySelector("#timer");

let socket = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let workletNode = null;
let readyResolver = null;
let flushResolver = null;
let sendingAudio = false;
let startedAt = 0;
let timerHandle = null;

function setStatus(kind, text) {
  statusDot.dataset.state = kind;
  statusText.textContent = text;
}

function setTranscript(text, isFinal = false) {
  transcript.textContent = text;
  transcript.classList.toggle("final", isFinal);
  placeholder.hidden = Boolean(text);
  const rtl = ["ar"].includes(languageSelect.value);
  transcript.dir = rtl ? "rtl" : "auto";
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function startTimer() {
  startedAt = performance.now();
  timer.textContent = "00:00";
  timerHandle = window.setInterval(() => {
    timer.textContent = formatDuration(performance.now() - startedAt);
  }, 250);
}

function stopTimer() {
  window.clearInterval(timerHandle);
  timerHandle = null;
}

function websocketURL() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/transcribe`;
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    socket = new WebSocket(websocketURL());
    socket.binaryType = "arraybuffer";
    readyResolver = resolve;

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        if (readyResolver) {
          readyResolver(message);
          readyResolver = null;
        }
        return;
      }
      if (message.type === "partial" || message.type === "final") {
        setTranscript(message.text, message.type === "final");
        const audioTime = Number(message.audio_seconds).toFixed(1);
        timing.textContent = `${audioTime}s audio · ${message.latency_ms}ms inference`;
        if (message.type === "final") {
          setStatus("done", "Transcription complete");
          finishUI();
        } else {
          setStatus("active", "Listening and transcribing");
        }
        return;
      }
      if (message.type === "error") {
        showError(message.message);
        void cleanupAudio();
      }
    };
    socket.onerror = () => reject(new Error("Could not connect to the QASR server"));
    socket.onclose = () => {
      sendingAudio = false;
      if (readyResolver) {
        readyResolver = null;
        reject(new Error("The QASR server closed the connection"));
      }
      if (!startButton.disabled) {
        return;
      }
      setStatus("error", "Connection to QASR closed");
      void cleanupAudio();
      finishUI();
    };
  });
}

async function setupMicrophone(targetSampleRate) {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  audioContext = new AudioContext({ latencyHint: "interactive" });
  await audioContext.audioWorklet.addModule("/static/audio-processor.js");
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "qasr-pcm16", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      targetSampleRate,
      chunkFrames: Math.round(targetSampleRate * 0.08),
    },
  });
  const silentOutput = audioContext.createGain();
  silentOutput.gain.value = 0;
  workletNode.connect(silentOutput).connect(audioContext.destination);
  workletNode.port.onmessage = (event) => {
    if (event.data.type === "pcm" && sendingAudio && socket?.readyState === WebSocket.OPEN) {
      socket.send(event.data.buffer);
    } else if (event.data.type === "level") {
      levelBar.style.transform = `scaleX(${event.data.value})`;
    } else if (event.data.type === "flushed" && flushResolver) {
      flushResolver();
      flushResolver = null;
    }
  };
  sourceNode.connect(workletNode);
  await audioContext.resume();
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    showError("This browser does not support AudioWorklet microphone capture.");
    return;
  }

  startButton.disabled = true;
  stopButton.disabled = true;
  languageSelect.disabled = true;
  transcript.classList.remove("final");
  timing.textContent = "Waiting for the first audio window…";
  setStatus("busy", "Requesting microphone access");

  try {
    const ready = await connectSocket();
    setStatus("busy", "Opening microphone");
    await setupMicrophone(ready.sample_rate);
    socket.send(JSON.stringify({
      type: "start",
      language: languageSelect.value || null,
    }));
    sendingAudio = true;
    stopButton.disabled = false;
    setStatus("active", "Listening");
    startTimer();
  } catch (error) {
    showError(error.message || String(error));
    await cleanupAudio();
    socket?.close();
    socket = null;
    finishUI();
  }
}

async function stopRecording() {
  if (!sendingAudio) {
    return;
  }
  stopButton.disabled = true;
  setStatus("busy", "Finalizing transcript");
  stopTimer();

  if (workletNode) {
    const flushed = new Promise((resolve) => {
      flushResolver = resolve;
    });
    workletNode.port.postMessage({ type: "flush" });
    await Promise.race([
      flushed,
      new Promise((resolve) => window.setTimeout(resolve, 200)),
    ]);
  }
  sendingAudio = false;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop" }));
  }
  await cleanupAudio();
}

async function cleanupAudio() {
  const activeSource = sourceNode;
  const activeWorklet = workletNode;
  const activeStream = mediaStream;
  const activeContext = audioContext;
  sourceNode = null;
  workletNode = null;
  mediaStream = null;
  audioContext = null;

  if (activeSource) {
    activeSource.disconnect();
  }
  if (activeWorklet) {
    activeWorklet.disconnect();
  }
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
  }
  if (activeContext && activeContext.state !== "closed") {
    await activeContext.close();
  }
  levelBar.style.transform = "scaleX(0)";
}

function finishUI() {
  stopTimer();
  startButton.disabled = false;
  stopButton.disabled = true;
  languageSelect.disabled = false;
}

function showError(message) {
  setStatus("error", message);
  timing.textContent = "Check the server log for details.";
  finishUI();
}

startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
clearButton.addEventListener("click", () => {
  setTranscript("");
  timing.textContent = "Results appear as you speak";
});

window.addEventListener("beforeunload", () => {
  mediaStream?.getTracks().forEach((track) => track.stop());
  socket?.close();
});
