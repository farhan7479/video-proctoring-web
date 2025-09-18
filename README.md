# 🚀 AI Proctoring System

A real-time video proctoring system with AI-powered detection for online interviews and exams.

## 🎯 Features

- **Real-time Face Detection** - OpenCV & MediaPipe powered face recognition
- **Focus Tracking** - Eye gaze and head pose monitoring  
- **Object Detection** - YOLOv8 detection of phones, books, suspicious items
- **Integrity Scoring** - Live scoring based on violations
- **Session Management** - Start, pause, stop monitoring sessions
- **Export Reports** - Download JSON, CSV, PDF reports
- **Modern UI** - Beautiful Tailwind CSS interface

## 📁 Project Structure

```
video-proctoring-web/
├── backend/          # FastAPI + AI Detection
│   ├── main.py       # FastAPI server
│   ├── detection/    # AI detection modules
│   ├── models/       # Data models
│   ├── utils/        # Utilities & logging
│   └── requirements.txt
├── frontend/         # Vite + Tailwind CSS
│   ├── src/
│   │   ├── main.js   # Main application
│   │   └── style.css # Tailwind styles
│   ├── index.html    # Main HTML
│   └── package.json
└── README.md
```

## 🛠️ Local Development Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** 
- **npm or yarn**
- **Webcam** (for testing)

### 🚀 Quick Start

#### Method 1: One Command (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd video-proctoring-web

# Kill any existing processes
pkill -f "python3 main.py" && pkill -f "vite"

# Start backend in background
cd backend && python3 main.py &

# Wait 3 seconds then start frontend
sleep 3 && cd ../frontend && npm run dev
```

#### Method 2: Separate Terminals

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
python3 main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

#### Method 3: Using Start Script

```bash
chmod +x start-dev.sh
./start-dev.sh
```

### 🔗 Local URLs

- **Frontend**: http://localhost:5173 (or port shown in terminal)
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 🛑 Stop Development Servers

```bash
pkill -f "python3 main.py" && pkill -f "vite"
```

## 🌐 Production Deployment

### Backend Deployment (Render)

1. **Push to GitHub:**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"
   - Configure:
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Root Directory**: `backend`

3. **Environment Variables:**
   - `PYTHON_VERSION`: `3.11.0`

4. **Get your backend URL**: `https://your-app-name.onrender.com`

### Frontend Deployment (Vercel)

1. **Set Environment Variable:**
   - In Vercel dashboard: `VITE_API_BASE` = `https://your-backend.onrender.com`

2. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Configure:
     - **Framework**: Vite
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist`

3. **Update Backend CORS:**
   - Add your Vercel domain to backend CORS settings
   - Redeploy backend

## 🔧 Configuration

### Environment Variables

**Frontend (.env.production):**
```env
VITE_API_BASE=https://your-backend.onrender.com
```

**Backend:**
```env
PORT=8000
PYTHON_VERSION=3.11.0
```

### Performance Settings

The system includes three performance modes:

- **Fast Mode**: 3-second processing intervals (best performance)
- **Balanced Mode**: 2-second intervals (default)
- **Accurate Mode**: 1-second intervals (best accuracy)

## 🎮 How to Use

### 1. Start a Session
- Enter candidate name
- Select interview type
- Grant camera permissions
- Click "Begin Interview"

### 2. Monitoring
- **Green indicators**: Normal behavior
- **Yellow indicators**: Minor issues
- **Red indicators**: Violations detected

### 3. Export Reports
After session ends, download:
- **JSON**: Complete session data
- **CSV**: Alert timeline
- **PDF**: Formatted report

## 🔍 AI Detection Features

### Face Detection
- **OpenCV Haar Cascades** (fallback)
- **MediaPipe** (when available)
- Detects presence and count of faces

### Object Detection  
- **YOLOv8** model for real-time detection
- Identifies: phones, books, laptops, keyboards
- Confidence-based filtering

### Gaze Tracking
- **Head pose estimation**
- **Eye tracking** (when MediaPipe available)
- **Focus status** determination

### Performance Optimizations
- Frame rate limiting (1-3 second intervals)
- Image compression and resizing
- Parallel AI processing
- Result caching

## 📊 API Documentation

### Main Endpoints

```
GET  /                      - Health check
GET  /health               - System status
GET  /sessions             - Active sessions
GET  /sessions/completed   - Session history
GET  /reports/{session_id} - Session report
GET  /export/{session_id}/{format} - Download report
WS   /ws/{session_id}      - Real-time monitoring
```

### WebSocket Events

**Send:**
```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,..."
}
```

**Receive:**
```json
{
  "face_count": 1,
  "focus_status": "good",
  "drowsiness_score": 0.1,
  "suspicious_count": 0,
  "integrity_score": 95
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Camera Access Denied**
```bash
# Grant camera permissions in browser settings
# Use HTTPS in production
```

**2. Backend Connection Failed**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check CORS settings
```

**3. WebSocket Connection Failed**
```bash
# Ensure WebSocket URL matches backend
# Check firewall settings
```

**4. Python Command Not Found (macOS)**
```bash
# Use python3 instead of python
which python3
python3 --version
```

**5. Missing Dependencies**
```bash
# Backend
pip install -r requirements.txt

# Frontend  
npm install
```

## 🧪 Testing

### Test Camera Access
```bash
# Frontend running at localhost:5173
# Click "Begin Interview" and grant permissions
```

### Test API Endpoints
```bash
curl http://localhost:8000/health
curl http://localhost:8000/sessions/completed
```

### Test Downloads
- Complete a session
- Try downloading JSON/CSV/PDF reports

## 📈 Performance Tips

1. **Optimize Frame Rate**: Use "Fast Mode" for better performance
2. **Close Other Apps**: Free up system resources
3. **Good Lighting**: Improves AI detection accuracy
4. **Stable Internet**: Required for WebSocket communication

## 🔒 Security Considerations

- **HTTPS Required**: For camera access in production
- **CORS Configuration**: Limit allowed origins
- **Data Privacy**: Session data stored locally, reports downloadable
- **No Audio Recording**: Only video processing

## 📝 Development Notes

### Adding New Detection Models
1. Create detector class in `backend/detection/`
2. Implement `initialize()` and `detect()` methods
3. Add to main processing pipeline
4. Update frontend to display results

### Customizing UI
- Modify `frontend/src/style.css` for styling
- Update `frontend/src/main.js` for functionality
- All styling uses Tailwind CSS classes

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push branch: `git push origin feature-name`
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check troubleshooting section above
2. Search existing GitHub issues
3. Create new issue with:
   - System information
   - Steps to reproduce
   - Error messages
   - Browser console logs

---

**Built with ❤️ using FastAPI, Vite, OpenCV, YOLOv8, and Tailwind CSS**