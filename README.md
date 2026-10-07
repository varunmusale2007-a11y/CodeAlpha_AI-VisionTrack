# AI-VisionTrack — Real-Time Object Detection & ByteTrack Intelligence

> **CodeAlpha Artificial Intelligence Internship — Task 4: Real-Time Object Detection and Tracking**  
> An enterprise-grade, full-stack Computer Vision web application combining **Ultralytics YOLOv8** for real-time object detection and **ByteTrack** for persistent multi-object tracking across in-browser live webcam streams.

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [CodeAlpha Task 4 Objective](#2-codealpha-task-4-objective)
3. [Key Features](#3-key-features)
4. [Technologies Used](#4-technologies-used)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [How YOLO Works in This Project](#7-how-yolo-works-in-this-project)
8. [How ByteTrack Works](#8-how-bytetrack-works)
9. [How Browser Webcam Communication Works](#9-how-browser-webcam-communication-works)
10. [Prerequisites & System Requirements](#10-prerequisites--system-requirements)
11. [Backend Installation](#11-backend-installation)
12. [Backend Startup Commands](#12-backend-startup-commands)
13. [Frontend Local Testing](#13-frontend-local-testing)
14. [How to Connect Frontend to Backend](#14-how-to-connect-frontend-to-backend)
15. [Deployment Architecture Overview](#15-deployment-architecture-overview)
16. [Vercel Frontend Deployment](#16-vercel-frontend-deployment)
17. [Python Backend Deployment (Render / Railway / Hugging Face)](#17-python-backend-deployment)
18. [How to Replace BACKEND_URL for Production](#18-how-to-replace-backend_url-for-production)
19. [Screenshots & Visual Demos](#19-screenshots--visual-demos)
20. [CodeAlpha Task Requirements Mapping](#20-codealpha-task-requirements-mapping)

---

## 1. Project Overview

**AI-VisionTrack** is a high-performance, real-time Computer Vision web application developed for the CodeAlpha Artificial Intelligence Internship (Task 4). It delivers real-time object localization, classification, and persistent identity tracking directly inside modern web browsers without requiring heavy client-side machine learning runtimes.

The frontend operates completely independently in the browser using HTML5, modern CSS3 with a futuristic dark cyber aesthetic, and Vanilla JavaScript. It captures webcam frames via standard WebRTC APIs (`navigator.mediaDevices.getUserMedia()`), downscales and compresses them to maintain sub-50ms latency, and transmits frames via asynchronous REST calls to a FastAPI backend powered by Ultralytics YOLOv8 and ByteTrack.

---

## 2. CodeAlpha Task 4 Objective

- **Goal**: Develop a robust object detection and tracking system capable of processing video or webcam feeds in real time.
- **Model Requirement**: Utilize a pretrained YOLO architecture for fast, accurate object recognition.
- **Tracking Requirement**: Implement ByteTrack to maintain consistent tracking IDs across sequential frames even through occlusions and rapid motion.
- **Display Requirement**: Render visual bounding boxes, object class labels, confidence scores (0–100%), and unique persistent track IDs dynamically on an HTML5 Canvas overlay.

---

## 3. Key Features

- ⚡ **Zero-Latency In-Browser Stream**: Ingests high-definition camera feeds via `navigator.mediaDevices.getUserMedia()` with custom aspect ratio scaling.
- 🎯 **YOLOv8 Detection**: Real-time multi-class object localization and classification across 80 COCO categories (people, vehicles, electronics, furniture, etc.).
- 🆔 **Stateful ByteTrack Multi-Object Tracking**: Preserves persistent tracking IDs across consecutive frames, matching both high- and low-confidence detection boxes to eliminate ID switching.
- 🎨 **Futuristic Cyber HUD**: Cyberpunk-inspired dark theme with glassmorphism cards, glowing bounding box brackets, target crosshairs, and dynamic motion trajectory trails.
- 📊 **Live Telemetry & Diagnostics**:
  - Processing FPS counter
  - Frame-level object counter
  - Active ByteTrack unique ID counter
  - Average confidence gauge
  - Server inference latency in milliseconds
- 🎛️ **Interactive Controls**:
  - Real-time confidence threshold slider (10% to 95%)
  - Request throttle rate regulator (2 to 15 FPS)
  - Visual display toggles (Boxes, IDs, Scores, Motion Trails)
  - Interactive ByteTrack state reset button
- 🌐 **Independent Full-Stack Decoupling**: Designed for independent frontend deployment on **Vercel** and backend deployment on **Render / Railway / Hugging Face Spaces**.

---

## 4. Technologies Used

### Frontend
- **HTML5**: Semantic layout and HTML5 `<video>` and `<canvas>` elements.
- **CSS3**: Custom design system with glassmorphism, responsive grid/flexbox layouts, CSS variables, and glowing neon keyframe animations.
- **Vanilla JavaScript (ES6+)**: Pure asynchronous JavaScript managing WebRTC video capture, off-screen canvas compression, REST API communication, and 60 FPS Canvas rendering.

### Backend
- **Python 3.10+**
- **FastAPI**: Asynchronous REST API framework.
- **Uvicorn**: High-performance ASGI web server.
- **Ultralytics YOLOv8**: Lightweight pretrained object detection model (`yolov8n.pt`).
- **ByteTrack**: High-performance multi-object tracker using Kalman filters and bipartite matching.
- **OpenCV (`cv2`)**: Image decoding, buffer parsing, and spatial coordinate normalization.
- **NumPy**: Matrix computations and tensor formatting.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT BROWSER                                │
│                                                                         │
│   ┌──────────────────────────┐         ┌────────────────────────────┐   │
│   │ navigator.mediaDevices   │         │ HTML5 Canvas Overlay       │   │
│   │ .getUserMedia() Webcam   │         │ Bounding Boxes + Track IDs │   │
│   └─────────────┬────────────┘         └─────────────▲──────────────┘   │
│                 │ (Raw Frames)                       │ (Render Loop)    │
│                 ▼                                    │                  │
│   ┌──────────────────────────┐         ┌─────────────┴──────────────┐   │
│   │ Offscreen Downscaling &  │         │ Detection List & Telemetry │   │
│   │ JPEG Quality Compression │         │ FPS / Latency / Counters   │   │
│   └─────────────┬────────────┘         └─────────────▲──────────────┘   │
│                 │                                    │                  │
└─────────────────┼────────────────────────────────────┼──────────────────┘
                  │ POST /api/detect                   │ JSON Response
                  │ (Base64 JPEG Frame)                │ (Detections + IDs)
                  ▼                                    │
┌──────────────────────────────────────────────────────┴──────────────────┐
│                          FASTAPI BACKEND                                │
│                                                                         │
│   ┌──────────────────────────┐         ┌────────────────────────────┐   │
│   │ OpenCV Image Decoder     │────────▶│ Ultralytics YOLOv8         │   │
│   │ (Base64 -> BGR ndarray)  │         │ (Feature & Box Extraction) │   │
│   └──────────────────────────┘         └─────────────┬──────────────┘   │
│                                                      │                  │
│                                                      ▼                  │
│                                        ┌────────────────────────────┐   │
│                                        │ ByteTrack Tracker State    │   │
│                                        │ (Kalman Filter + Track IDs)│   │
│                                        └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Project Structure

```
AI-VisionTrack/
│
├── frontend/
│   ├── index.html          # Semantic HTML5 UI layout and HUD overlay
│   ├── style.css           # Futuristic dark cyber design system
│   ├── app.js              # Camera capture, REST client, and canvas renderer
│   └── assets/
│       └── logo.svg        # High-tech vector aperture logo
│
├── backend/
│   ├── main.py             # FastAPI server with CORS, health, & detect endpoints
│   ├── detector.py         # YOLO model manager and detection pipeline
│   ├── tracker.py          # ByteTrack multi-object tracking state manager
│   ├── requirements.txt    # Python dependencies
│   └── models/             # Local cache for YOLO weights (yolov8n.pt)
│
└── README.md               # Full-stack documentation and deployment guide
```

---

## 7. How YOLO Works in This Project

1. **Model Initialization**: The backend loads `yolov8n.pt` (a lightweight, highly optimized nano model) into memory once during application startup, running a warmup frame to eliminate first-request latency.
2. **Single-Stage Detection**: When a compressed frame is received at `/api/detect`, YOLO performs single-pass feature extraction across multiple anchor-free scales.
3. **Bounding Box & Class Output**: YOLO predicts `[x1, y1, x2, y2]` coordinates, class indices (e.g. `person`, `laptop`, `cell phone`), and confidence scores for every candidate object exceeding the user's selected threshold.

---

## 8. How ByteTrack Works

Traditional object trackers discard low-confidence detection boxes to reduce false positives, which causes frequent ID switching when objects are occluded or blurred. **ByteTrack** solves this by preserving both high- and low-score boxes:

1. **Stateful Kalman Filtering**: Maintains motion velocity and position estimates for all existing track IDs across consecutive frames.
2. **First Association**: High-confidence detections are matched to existing active tracks using IoU (Intersection over Union) distance and Hungarian bipartite matching.
3. **Second Association**: Unmatched active tracks are then matched against **low-confidence** detections, preserving trajectories through temporary occlusions.
4. **Track Persistence**: In `backend/tracker.py`, the ByteTrack instance retains its Kalman filter state across consecutive frame requests (`persist=True`), ensuring object IDs remain stable throughout the live session.

---

## 9. How Browser Webcam Communication Works

- **Direct In-Browser Access**: The frontend calls `navigator.mediaDevices.getUserMedia({ video: true })` inside the user's browser.
- **No Backend Video Capture**: The backend **never** calls `cv2.VideoCapture(0)` (which would fail on remote cloud servers). The browser manages the local camera hardware directly.
- **Client-Side Optimization**:
  - An off-screen canvas resizes raw video frames to a maximum width of 640px.
  - The frame is compressed to JPEG quality (~0.65).
  - The compressed image is encoded as a Base64 string and dispatched to the backend via `fetch()`.
- **Concurrency Control**: JavaScript uses an `inFlightRequest` guard to ensure frames are only sent after the previous detection roundtrip completes, preventing queue build-up and keeping the UI responsive at 60 FPS.

---

## 10. Prerequisites & System Requirements

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- **Node.js / Live Server** (Optional for local frontend hosting, e.g. Python `http.server`, VS Code Live Server, or `npx serve`)
- **Webcam**: Built-in or external USB camera
- **Supported Browsers**: Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari

---

## 11. Backend Installation

1. Open your terminal and navigate to the `backend` folder:
   ```bash
   cd AI-VisionTrack/backend
   ```

2. (Optional but recommended) Create and activate a Python virtual environment:
   - **Windows**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 12. Backend Startup Commands

Start the FastAPI application with Uvicorn:

```bash
python main.py
```
*Alternatively:*
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

When started, the backend will output:
```text
[System] Loading YOLO and ByteTrack models...
[Detector] Model warmup complete.
[Tracker] ByteTrack tracker warmed up and ready.
[System] Models initialized successfully. Backend ready.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verify backend health in your browser at:  
👉 **`http://localhost:8000/api/health`**

---

## 13. Frontend Local Testing

To run the frontend locally:

### Option A: Python Built-in HTTP Server (Recommended)
Open a new terminal in the `frontend` folder:
```bash
cd AI-VisionTrack/frontend
python -m http.server 3000
```
Open your browser and navigate to: **`http://localhost:3000`**

### Option B: Node.js Serve / Live Server
```bash
npx serve AI-VisionTrack/frontend -p 3000
```

### Option C: Direct File Opening
You can also open `frontend/index.html` directly in modern web browsers (Chrome / Edge / Firefox).

---

## 14. How to Connect Frontend to Backend

1. Ensure the Python backend is running on `http://localhost:8000`.
2. Open the frontend in your browser.
3. The top navigation bar will automatically display `Online: YOLOv8n + ByteTrack` with a glowing green indicator.
4. If your backend is hosted on a custom port or remote server, click the **Gear (⚙️) icon** in the header or update `BACKEND_URL` in `frontend/app.js`.
5. Click **Start Camera**, grant webcam permissions, and click **Start Detection**.

---

## 15. Deployment Architecture Overview

```
┌────────────────────────┐                   ┌────────────────────────┐
│     Vercel Edge        │   HTTPS REST API  │ Python Cloud Host      │
│  Static Frontend       │ ────────────────▶ │ (Render / Railway / HF)│
│ (HTML5/CSS3/Vanilla JS)│ ◀──────────────── │ (FastAPI + YOLOv8)     │
└────────────────────────┘                   └────────────────────────┘
```

The application is completely decoupled:
- **Frontend** contains zero Python runtime dependencies and can be hosted statically for free on Vercel, Netlify, or GitHub Pages.
- **Backend** runs FastAPI with PyTorch, Ultralytics, and OpenCV on any Python hosting provider.

---

## 16. Vercel Frontend Deployment

1. Push your repository to **GitHub**.
2. Log into [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. In the project settings:
   - **Root Directory**: Select `frontend`
   - **Framework Preset**: Select `Other`
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty (serves root `index.html`)
5. Click **Deploy**. Vercel will generate your live production URL (e.g. `https://ai-visiontrack.vercel.app`).

---

## 17. Python Backend Deployment

Deploy the `backend` folder to any Python-supported hosting service:

### Option 1: Render (Web Service)
1. Create a new **Web Service** on [Render.com](https://render.com).
2. Connect your GitHub repository and set **Root Directory** to `backend`.
3. Set **Runtime** to `Python 3`.
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click **Create Web Service**.

### Option 2: Railway
1. Create a new project on [Railway.app](https://railway.app).
2. Deploy from GitHub repo with `backend` as the root directory.
3. Railway automatically detects `requirements.txt` and starts `uvicorn main:app --host 0.0.0.0 --port $PORT`.

### Option 3: Hugging Face Spaces (Docker / Gradio / FastAPI)
1. Create a new **Space** on [Hugging Face](https://huggingface.co/spaces) with SDK: `Docker` or `FastAPI`.
2. Upload the `backend/` files and expose port `7860`.

---

## 18. How to Replace BACKEND_URL for Production

Once your Python backend is deployed and you have its public URL (e.g., `https://ai-visiontrack-api.onrender.com`):

### Method 1: Using the In-App Settings UI (Instant)
1. Open your deployed Vercel frontend.
2. Click the **Gear (⚙️) Settings button** in the top-right header.
3. Enter your deployed backend URL: `https://ai-visiontrack-api.onrender.com`
4. Click **Save & Test**. The URL is saved in your browser's `localStorage`.

### Method 2: Permanent Code Replacement
Edit line 14 of `frontend/app.js`:

```javascript
// Replace default URL with your production backend endpoint
const DEFAULT_BACKEND_URL = "https://ai-visiontrack-api.onrender.com";
```

---

## 19. Screenshots & Visual Demos

```
+-------------------------------------------------------------------------+
| [AI] AI-VISIONTRACK  (CodeAlpha Task 4)    [● Online: YOLOv8n] [⚙ Settings] |
+-------------------------------------------------------------------------+
|                                                                         |
|                       REAL-TIME OBJECT INTELLIGENCE                     |
|         [YOLOv8]  [ByteTrack]  [OpenCV]  [Computer Vision]  [FastAPI]   |
|                                                                         |
|  +-------------------------------------+  +--------------------------+  |
|  | LIVE CAMERA FEED        640x480 32ms |  | INFERENCE SETTINGS       |  |
|  | +---------------------------------+ |  | Confidence: [===o====] 35%|  |
|  | |                                 | |  | Request Rate: [==o====] 8FPS |  |
|  | |    +---------------+            | |  |                          |  |
|  | |    | PERSON | ID:3 | 94%        | |  | [X] Boxes  [X] Track IDs |  |
|  | |    | [•]           |            | |  | [X] Scores [X] Trails    |  |
|  | |    +---------------+            | |  +--------------------------+  |
|  | |                                 | |  | TRACKED OBJECTS FEED     |  |
|  | +---------------------------------+ |  | [ID #3] Person       94% |  |
|  | [Start Camera] [Stop] [Start Detect]|  | [ID #7] Laptop       88% |  |
|  +-------------------------------------+  +--------------------------+  |
|                                                                         |
|  [FPS: 28.5]    [TOTAL: 2]    [TRACKED IDS: 2]    [AVG CONF: 91%]       |
+-------------------------------------------------------------------------+
```

---

## 20. CodeAlpha Task Requirements Mapping

| Requirement Specified in Task 4 | Implementation in AI-VisionTrack | Status |
| :--- | :--- | :---: |
| **Real-time object detection** | Ultralytics YOLOv8 inference running on server stream | ✅ Complete |
| **Object tracking with unique IDs** | ByteTrack Kalman filter tracking state in `backend/tracker.py` | ✅ Complete |
| **Webcam / Video input** | In-browser WebRTC `navigator.mediaDevices.getUserMedia()` | ✅ Complete |
| **Bounding box visualization** | High-tech canvas overlay with corner brackets & glow | ✅ Complete |
| **Confidence score display** | Dynamic percentage badges on boxes and live feed | ✅ Complete |
| **Tracking ID display** | Persistent `ID: #N` badges mapped with distinct colors | ✅ Complete |
| **Decoupled REST API backend** | FastAPI server with `/api/health`, `/api/detect`, `/api/reset-tracker` | ✅ Complete |
| **No server-side webcam access** | 100% browser-managed camera stream via HTTP POST payloads | ✅ Complete |
| **Production-ready UI** | Responsive dark cyber interface with live telemetry | ✅ Complete |

---

## 👨‍💻 Author & Internship Details
- **Project**: AI-VisionTrack (Task 4)
- **Internship**: CodeAlpha Artificial Intelligence Internship
- **Role**: AI Engineering Intern
