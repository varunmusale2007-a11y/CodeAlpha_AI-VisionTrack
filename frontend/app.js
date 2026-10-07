/**
 * ==============================================================================
 * AI-VisionTrack - Frontend Application Logic
 * CodeAlpha Artificial Intelligence Internship Task 4
 * ==============================================================================
 * 
 * Real-time object detection and persistent multi-object tracking dashboard.
 * Communicates with the FastAPI backend running YOLOv8 and ByteTrack.
 */

// ============================================================================
// BACKEND CONFIGURATION
// Replace this URL with your deployed backend URL (e.g. Render/Railway/Fly.io)
// ============================================================================
const DEFAULT_BACKEND_URL = "http://localhost:8000";
let BACKEND_URL = localStorage.getItem("ai_visiontrack_backend_url") || DEFAULT_BACKEND_URL;

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const state = {
  isCameraRunning: false,
  isDetecting: false,
  stream: null,
  detectionTimer: null,
  inFlightRequest: false,
  confidenceThreshold: 0.35,
  targetFps: 8,
  backendConnected: false,
  
  // Visual Toggles
  showBoxes: true,
  showIds: true,
  showScores: true,
  showTrails: true,
  
  // Tracking history for motion trails: Map<track_id, Array<{x, y, timestamp}>>
  trackHistories: new Map(),
  
  // Performance Telemetry
  lastFrameTime: performance.now(),
  fpsFrames: 0,
  currentFps: 0,
  lastInferenceMs: 0,
  
  // Active Detections Cache
  currentDetections: [],
  sourceFrameWidth: 640,
  sourceFrameHeight: 480
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================
const elements = {
  // Video & Canvas
  webcamVideo: document.getElementById("webcamVideo"),
  detectionCanvas: document.getElementById("detectionCanvas"),
  standbyOverlay: document.getElementById("standbyOverlay"),
  liveDot: document.getElementById("liveDot"),
  feedTitle: document.getElementById("feedTitle"),
  resolutionBadge: document.getElementById("resolutionBadge"),
  streamLatencyBadge: document.getElementById("streamLatencyBadge"),
  
  // Buttons
  startCamBtn: document.getElementById("startCamBtn"),
  stopCamBtn: document.getElementById("stopCamBtn"),
  startDetectBtn: document.getElementById("startDetectBtn"),
  stopDetectBtn: document.getElementById("stopDetectBtn"),
  resetTrackerBtn: document.getElementById("resetTrackerBtn"),
  quickStartBtn: document.getElementById("quickStartBtn"),
  clearFeedBtn: document.getElementById("clearFeedBtn"),
  
  // Telemetry Counters
  fpsCounter: document.getElementById("fpsCounter"),
  totalObjectsCounter: document.getElementById("totalObjectsCounter"),
  trackedObjectsCounter: document.getElementById("trackedObjectsCounter"),
  avgConfidence: document.getElementById("avgConfidence"),
  
  // Status & Alerts
  backendStatusBadge: document.getElementById("backendStatusBadge"),
  backendStatusText: document.getElementById("backendStatusText"),
  alertBanner: document.getElementById("alertBanner"),
  alertMessage: document.getElementById("alertMessage"),
  closeAlertBtn: document.getElementById("closeAlertBtn"),
  
  // Feeds & Logs
  objectsListContainer: document.getElementById("objectsListContainer"),
  emptyFeedMessage: document.getElementById("emptyFeedMessage"),
  feedCountBadge: document.getElementById("feedCountBadge"),
  consoleOutput: document.getElementById("consoleOutput"),
  
  // Settings & Toggles
  confSlider: document.getElementById("confSlider"),
  confValText: document.getElementById("confValText"),
  fpsThrottleSlider: document.getElementById("fpsThrottleSlider"),
  fpsThrottleText: document.getElementById("fpsThrottleText"),
  toggleBoxes: document.getElementById("toggleBoxes"),
  toggleIds: document.getElementById("toggleIds"),
  toggleScores: document.getElementById("toggleScores"),
  toggleTrails: document.getElementById("toggleTrails"),
  
  // Config Modal
  configToggleBtn: document.getElementById("configToggleBtn"),
  configModal: document.getElementById("configModal"),
  closeConfigBtn: document.getElementById("closeConfigBtn"),
  backendUrlInput: document.getElementById("backendUrlInput"),
  saveConfigBtn: document.getElementById("saveConfigBtn")
};

const canvasCtx = elements.detectionCanvas.getContext("2d");

// Dedicated Off-screen Canvas for Frame Capture & Downscaling
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

// Distinct Vibrant Cyber Color Palette for Track IDs
const TRACK_COLORS = [
  "#00f2fe", // Cyber Cyan
  "#10b981", // Emerald Neon
  "#f59e0b", // Amber Gold
  "#a855f7", // Purple Glow
  "#ec4899", // Neon Pink
  "#3b82f6", // Electric Blue
  "#14b8a6", // Bright Teal
  "#f43f5e", // Rose Red
  "#84cc16", // Lime
  "#06b6d4"  // Sky Cyan
];

function getTrackColor(trackId) {
  const id = Math.abs(parseInt(trackId, 10) || 0);
  return TRACK_COLORS[id % TRACK_COLORS.length];
}

// ============================================================================
// LOGGING & ALERTS
// ============================================================================
function log(message, type = "detect") {
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0];
  
  const entry = document.createElement("div");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">[${timeStr}]</span> ${message}`;
  
  elements.consoleOutput.appendChild(entry);
  elements.consoleOutput.scrollTop = elements.consoleOutput.scrollHeight;
  
  // Keep log size bounded
  while (elements.consoleOutput.children.length > 50) {
    elements.consoleOutput.removeChild(elements.consoleOutput.firstChild);
  }
}

function showAlert(message) {
  elements.alertMessage.textContent = message;
  elements.alertBanner.classList.remove("hidden");
}

function hideAlert() {
  elements.alertBanner.classList.add("hidden");
}

// ============================================================================
// BACKEND HEALTH CHECK & CONNECTIVITY
// ============================================================================
async function checkBackendHealth() {
  elements.backendStatusBadge.className = "status-badge status-checking";
  elements.backendStatusText.textContent = "Checking Backend...";
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);
    
    const response = await fetch(`${BACKEND_URL}/api/health`, {
      method: "GET",
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      state.backendConnected = true;
      elements.backendStatusBadge.className = "status-badge status-online";
      elements.backendStatusText.textContent = `Online: ${data.model || "YOLOv8"} + ByteTrack`;
      log(`Backend Connected: ${data.service || "AI-VisionTrack"} (v${data.version || "1.0.0"})`, "success");
      hideAlert();
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    state.backendConnected = false;
    elements.backendStatusBadge.className = "status-badge status-offline";
    elements.backendStatusText.textContent = "Backend Offline";
    log(`Backend Health Check Failed: ${error.message}. Is main.py running at ${BACKEND_URL}?`, "warning");
  }
}

// ============================================================================
// WEBCAM STREAM CONTROLLER (navigator.mediaDevices.getUserMedia)
// ============================================================================
async function startCamera() {
  hideAlert();
  log("Requesting user webcam access...", "system");
  
  try {
    // WebRTC getUserMedia standard constraints
    const constraints = {
      video: {
        width: { ideal: 1280, max: 1920 },
        height: { ideal: 720, max: 1080 },
        facingMode: "user"
      },
      audio: false
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    elements.webcamVideo.srcObject = stream;
    
    await new Promise((resolve) => {
      elements.webcamVideo.onloadedmetadata = () => {
        elements.webcamVideo.play();
        resolve();
      };
    });
    
    state.isCameraRunning = true;
    
    // UI Updates
    elements.standbyOverlay.classList.add("hidden");
    elements.liveDot.classList.add("active");
    elements.feedTitle.textContent = "LIVE CAMERA FEED";
    elements.startCamBtn.disabled = true;
    elements.stopCamBtn.disabled = false;
    elements.startDetectBtn.disabled = false;
    
    // Match Canvas dimensions to video
    syncCanvasSize();
    
    const vWidth = elements.webcamVideo.videoWidth || 640;
    const vHeight = elements.webcamVideo.videoHeight || 480;
    elements.resolutionBadge.textContent = `${vWidth} \u00D7 ${vHeight}`;
    
    log(`Webcam stream active (${vWidth}x${vHeight}). Ready for detection.`, "success");
    
    // Start continuous rendering loop
    requestAnimationFrame(renderCanvasLoop);
  } catch (err) {
    console.error("Camera access error:", err);
    state.isCameraRunning = false;
    let errMsg = "Unable to access webcam.";
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      errMsg = "Webcam permission was denied. Please allow camera access in your browser settings.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      errMsg = "No camera device found on this system.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      errMsg = "Webcam is already in use by another application.";
    }
    showAlert(errMsg);
    log(`Camera Error: ${errMsg}`, "error");
  }
}

function stopCamera() {
  if (state.isDetecting) {
    stopDetection();
  }
  
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }
  
  elements.webcamVideo.srcObject = null;
  state.isCameraRunning = false;
  
  // UI Updates
  elements.standbyOverlay.classList.remove("hidden");
  elements.liveDot.classList.remove("active");
  elements.feedTitle.textContent = "FEED STANDBY";
  elements.startCamBtn.disabled = false;
  elements.stopCamBtn.disabled = true;
  elements.startDetectBtn.disabled = true;
  elements.stopDetectBtn.disabled = true;
  
  // Clear Canvas and counters
  canvasCtx.clearRect(0, 0, elements.detectionCanvas.width, elements.detectionCanvas.height);
  state.currentDetections = [];
  state.trackHistories.clear();
  updateTelemetryUI([], 0);
  
  log("Webcam stream stopped.", "system");
}

function syncCanvasSize() {
  const rect = elements.webcamVideo.getBoundingClientRect();
  const width = elements.webcamVideo.videoWidth || rect.width || 640;
  const height = elements.webcamVideo.videoHeight || rect.height || 480;
  
  if (elements.detectionCanvas.width !== width || elements.detectionCanvas.height !== height) {
    elements.detectionCanvas.width = width;
    elements.detectionCanvas.height = height;
  }
}

// Resize canvas when window changes
window.addEventListener("resize", () => {
  if (state.isCameraRunning) {
    syncCanvasSize();
  }
});

// ============================================================================
// FRAME CAPTURE & COMPRESSION
// ============================================================================
function captureCompressedFrame() {
  if (!state.isCameraRunning || elements.webcamVideo.readyState < 2) {
    return null;
  }
  
  const video = elements.webcamVideo;
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  
  // Downscale frame to max width 640px to ensure fast network transport & low latency
  const targetWidth = Math.min(640, vw);
  const targetHeight = Math.round((targetWidth / vw) * vh);
  
  captureCanvas.width = targetWidth;
  captureCanvas.height = targetHeight;
  
  captureCtx.drawImage(video, 0, 0, targetWidth, targetHeight);
  
  // Compress to JPEG with 0.65 quality
  return {
    dataUrl: captureCanvas.toDataURL("image/jpeg", 0.65),
    width: targetWidth,
    height: targetHeight
  };
}

// ============================================================================
// DETECTION & TRACKING LOOP
// ============================================================================
function startDetection() {
  if (!state.isCameraRunning) {
    showAlert("Please start the camera before starting detection.");
    return;
  }
  
  state.isDetecting = true;
  elements.startDetectBtn.disabled = true;
  elements.stopDetectBtn.disabled = false;
  hideAlert();
  log(`Object detection & ByteTrack tracking started at ${state.targetFps} FPS target.`, "success");
  
  // Schedule frame transmissions
  scheduleNextDetection();
}

function stopDetection() {
  state.isDetecting = false;
  if (state.detectionTimer) {
    clearTimeout(state.detectionTimer);
    state.detectionTimer = null;
  }
  
  elements.startDetectBtn.disabled = !state.isCameraRunning;
  elements.stopDetectBtn.disabled = true;
  
  state.currentDetections = [];
  updateTelemetryUI([], 0);
  log("Object detection stopped.", "system");
}

function scheduleNextDetection() {
  if (!state.isDetecting) return;
  
  const intervalMs = Math.max(50, Math.round(1000 / state.targetFps));
  state.detectionTimer = setTimeout(async () => {
    if (state.isDetecting) {
      await processDetectionFrame();
      scheduleNextDetection();
    }
  }, intervalMs);
}

async function processDetectionFrame() {
  // Prevent sending frames while previous request is still in-flight
  if (state.inFlightRequest || !state.isDetecting || !state.isCameraRunning) {
    return;
  }
  
  const frame = captureCompressedFrame();
  if (!frame) return;
  
  state.inFlightRequest = true;
  const startTime = performance.now();
  
  try {
    const payload = {
      image: frame.dataUrl,
      confidence: state.confidenceThreshold,
      use_tracker: true
    };
    
    const response = await fetch(`${BACKEND_URL}/api/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    const roundtripMs = Math.round(performance.now() - startTime);
    elements.streamLatencyBadge.textContent = `${roundtripMs} ms`;
    
    if (response.ok) {
      const data = await response.json();
      state.sourceFrameWidth = (data.metadata && data.metadata.frame_width) || frame.width;
      state.sourceFrameHeight = (data.metadata && data.metadata.frame_height) || frame.height;
      
      const detections = data.detections || [];
      state.currentDetections = detections;
      
      // Update trajectory histories for tracked IDs
      updateTrackHistories(detections);
      
      // Update UI Telemetry & Feed
      updateTelemetryUI(detections, roundtripMs);
      updateObjectsFeed(detections);
    } else {
      const errorText = await response.text();
      log(`Detection API error (${response.status}): ${errorText.substring(0, 100)}`, "warning");
    }
  } catch (err) {
    log(`Network request failed: ${err.message}`, "error");
    state.backendConnected = false;
    elements.backendStatusBadge.className = "status-badge status-offline";
    elements.backendStatusText.textContent = "Backend Disconnected";
  } finally {
    state.inFlightRequest = false;
  }
}

// ============================================================================
// TRACK HISTORY & TRAJECTORY TRAILS
// ============================================================================
function updateTrackHistories(detections) {
  const currentTrackIds = new Set();
  const now = Date.now();
  
  detections.forEach(det => {
    if (det.track_id !== undefined && det.track_id !== null) {
      const trackId = det.track_id;
      currentTrackIds.add(trackId);
      
      const [x1, y1, x2, y2] = det.bbox;
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      
      if (!state.trackHistories.has(trackId)) {
        state.trackHistories.set(trackId, []);
      }
      
      const history = state.trackHistories.get(trackId);
      history.push({ x: centerX, y: centerY, time: now });
      
      // Keep only recent 20 points
      if (history.length > 20) {
        history.shift();
      }
    }
  });
  
  // Clean up stale track histories older than 5 seconds
  for (const [id, history] of state.trackHistories.entries()) {
    if (!currentTrackIds.has(id)) {
      if (history.length === 0 || (now - history[history.length - 1].time > 5000)) {
        state.trackHistories.delete(id);
      }
    }
  }
}

// ============================================================================
// CANVAS RENDERING (HIGH-TECH CYBER HUD & BOUNDING BOXES)
// ============================================================================
function renderCanvasLoop(timestamp) {
  if (state.isCameraRunning) {
    // Calculate FPS
    state.fpsFrames++;
    if (timestamp - state.lastFrameTime >= 1000) {
      state.currentFps = (state.fpsFrames * 1000) / (timestamp - state.lastFrameTime);
      elements.fpsCounter.textContent = state.currentFps.toFixed(1);
      state.fpsFrames = 0;
      state.lastFrameTime = timestamp;
    }
    
    drawOverlay();
  }
  
  requestAnimationFrame(renderCanvasLoop);
}

function drawOverlay() {
  const canvas = elements.detectionCanvas;
  const ctx = canvasCtx;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (!state.isDetecting || state.currentDetections.length === 0) {
    return;
  }
  
  const scaleX = canvas.width / state.sourceFrameWidth;
  const scaleY = canvas.height / state.sourceFrameHeight;
  
  // 1. Draw Motion Trails
  if (state.showTrails) {
    state.trackHistories.forEach((history, trackId) => {
      if (history.length < 2) return;
      
      const color = getTrackColor(trackId);
      ctx.beginPath();
      ctx.moveTo(history[0].x * scaleX, history[0].y * scaleY);
      
      for (let i = 1; i < history.length; i++) {
        ctx.lineTo(history[i].x * scaleX, history[i].y * scaleY);
      }
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.globalAlpha = 0.6;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
    });
  }
  
  // 2. Draw Detections Bounding Boxes & Badges
  state.currentDetections.forEach(det => {
    const [origX1, origY1, origX2, origY2] = det.bbox;
    
    const x1 = origX1 * scaleX;
    const y1 = origY1 * scaleY;
    const x2 = origX2 * scaleX;
    const y2 = origY2 * scaleY;
    
    const w = x2 - x1;
    const h = y2 - y1;
    
    const color = getTrackColor(det.track_id);
    const labelClass = det.class || "object";
    const confidencePct = Math.round((det.confidence || 0) * 100);
    const trackId = det.track_id;
    
    // Draw Bounding Box
    if (state.showBoxes) {
      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      
      // Box fill
      ctx.fillStyle = `${color}18`;
      ctx.fillRect(x1, y1, w, h);
      
      // Main border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);
      
      // Cyber Corner Brackets
      const cornerLen = Math.min(18, w / 4, h / 4);
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "#ffffff";
      
      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLen);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerLen, y1);
      ctx.stroke();
      
      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerLen);
      ctx.stroke();
      
      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + h - cornerLen);
      ctx.lineTo(x1, y1 + h);
      ctx.lineTo(x1 + cornerLen, y1 + h);
      ctx.stroke();
      
      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y1 + h);
      ctx.lineTo(x2, y1 + h);
      ctx.lineTo(x2, y1 + h - cornerLen);
      ctx.stroke();
      
      // Center Target Reticle
      const cx = x1 + w / 2;
      const cy = y1 + h / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      
      ctx.shadowBlur = 0;
    }
    
    // Draw Label Pill (Class + Track ID + Confidence)
    if (state.showIds || state.showScores) {
      ctx.font = "bold 12px 'Chakra Petch', 'JetBrains Mono', sans-serif";
      
      let labelText = labelClass.toUpperCase();
      if (state.showIds && trackId !== undefined) {
        labelText += ` | ID:${trackId}`;
      }
      if (state.showScores) {
        labelText += ` | ${confidencePct}%`;
      }
      
      const textMetrics = ctx.measureText(labelText);
      const textW = textMetrics.width;
      const pillHeight = 22;
      const pillWidth = textW + 16;
      
      // Position pill above box (or inside if near top edge)
      let pillX = x1;
      let pillY = y1 - pillHeight - 4;
      if (pillY < 0) {
        pillY = y1 + 4;
      }
      
      // Pill Background
      ctx.fillStyle = "#030712dd";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillWidth, pillHeight, 4);
      ctx.fill();
      ctx.stroke();
      
      // Pill Text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, pillX + 8, pillY + 15);
    }
  });
}

// ============================================================================
// TELEMETRY & FEED UI UPDATES
// ============================================================================
function updateTelemetryUI(detections, roundtripMs) {
  const total = detections.length;
  elements.totalObjectsCounter.textContent = total;
  
  // Count unique active track IDs
  const uniqueTrackIds = new Set(detections.map(d => d.track_id).filter(id => id !== undefined));
  elements.trackedObjectsCounter.textContent = uniqueTrackIds.size;
  
  if (total > 0) {
    const avgConf = detections.reduce((sum, d) => sum + (d.confidence || 0), 0) / total;
    elements.avgConfidence.textContent = `${Math.round(avgConf * 100)}%`;
  } else {
    elements.avgConfidence.textContent = "0%";
  }
}

function updateObjectsFeed(detections) {
  elements.feedCountBadge.textContent = `${detections.length} Detected`;
  
  if (detections.length === 0) {
    elements.emptyFeedMessage.style.display = "block";
    elements.objectsListContainer.querySelectorAll(".object-item").forEach(el => el.remove());
    return;
  }
  
  elements.emptyFeedMessage.style.display = "none";
  elements.objectsListContainer.querySelectorAll(".object-item").forEach(el => el.remove());
  
  detections.forEach(det => {
    const trackId = det.track_id !== undefined ? det.track_id : "-";
    const color = getTrackColor(trackId);
    const confPct = Math.round((det.confidence || 0) * 100);
    
    const item = document.createElement("div");
    item.className = "object-item";
    item.style.borderLeftColor = color;
    
    item.innerHTML = `
      <div class="object-item-left">
        <span class="track-id-badge" style="color: ${color}; border-color: ${color}55; background: ${color}15;">
          ID #${trackId}
        </span>
        <span class="object-class-name">${det.class || "Unknown"}</span>
      </div>
      <div class="object-item-right">
        <span class="object-conf-pill">${confPct}%</span>
        <div class="conf-bar-wrap">
          <div class="conf-bar-fill" style="width: ${confPct}%; background: ${color};"></div>
        </div>
      </div>
    `;
    
    elements.objectsListContainer.appendChild(item);
  });
}

// ============================================================================
// RESET TRACKER HANDLER
// ============================================================================
async function resetTracker() {
  state.trackHistories.clear();
  try {
    const res = await fetch(`${BACKEND_URL}/api/reset-tracker`, { method: "POST" });
    if (res.ok) {
      log("ByteTrack tracking IDs reset successfully.", "system");
    }
  } catch (e) {
    log(`Tracker reset failed: ${e.message}`, "warning");
  }
}

// ============================================================================
// EVENT LISTENERS & INITIALIZATION
// ============================================================================
function setupEventListeners() {
  // Primary Control Buttons
  elements.startCamBtn.addEventListener("click", startCamera);
  elements.quickStartBtn.addEventListener("click", startCamera);
  elements.stopCamBtn.addEventListener("click", stopCamera);
  elements.startDetectBtn.addEventListener("click", startDetection);
  elements.stopDetectBtn.addEventListener("click", stopDetection);
  elements.resetTrackerBtn.addEventListener("click", resetTracker);
  
  // Clear Feed
  elements.clearFeedBtn.addEventListener("click", () => {
    elements.objectsListContainer.querySelectorAll(".object-item").forEach(el => el.remove());
    elements.emptyFeedMessage.style.display = "block";
    elements.feedCountBadge.textContent = "0 Detected";
  });
  
  // Close Alert
  elements.closeAlertBtn.addEventListener("click", hideAlert);
  
  // Sliders
  elements.confSlider.addEventListener("input", (e) => {
    state.confidenceThreshold = parseFloat(e.target.value);
    elements.confValText.textContent = `${Math.round(state.confidenceThreshold * 100)}%`;
  });
  
  elements.fpsThrottleSlider.addEventListener("input", (e) => {
    state.targetFps = parseInt(e.target.value, 10);
    elements.fpsThrottleText.textContent = `${state.targetFps} FPS`;
  });
  
  // Visual Toggles
  elements.toggleBoxes.addEventListener("change", (e) => { state.showBoxes = e.target.checked; });
  elements.toggleIds.addEventListener("change", (e) => { state.showIds = e.target.checked; });
  elements.toggleScores.addEventListener("change", (e) => { state.showScores = e.target.checked; });
  elements.toggleTrails.addEventListener("change", (e) => { state.showTrails = e.target.checked; });
  
  // Config Modal
  elements.configToggleBtn.addEventListener("click", () => {
    elements.backendUrlInput.value = BACKEND_URL;
    elements.configModal.classList.remove("hidden");
  });
  
  elements.closeConfigBtn.addEventListener("click", () => {
    elements.configModal.classList.add("hidden");
  });
  
  elements.configModal.addEventListener("click", (e) => {
    if (e.target === elements.configModal) {
      elements.configModal.classList.add("hidden");
    }
  });
  
  elements.saveConfigBtn.addEventListener("click", () => {
    const newUrl = elements.backendUrlInput.value.trim().replace(/\/+$/, "");
    if (newUrl) {
      BACKEND_URL = newUrl;
      localStorage.setItem("ai_visiontrack_backend_url", newUrl);
      elements.configModal.classList.add("hidden");
      log(`Backend endpoint updated to: ${BACKEND_URL}`, "system");
      checkBackendHealth();
    }
  });
}

// Initial Boot
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  checkBackendHealth();
  
  // Periodic backend health ping every 10 seconds
  setInterval(checkBackendHealth, 10000);
});
