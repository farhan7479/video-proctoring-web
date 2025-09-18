from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
import cv2
import numpy as np
import base64
import json
import asyncio
from typing import Dict, List
import logging
from datetime import datetime
import uuid

from detection.face_detector import FaceDetector
from detection.object_detector import ObjectDetector  
from detection.gaze_detector import GazeDetector
from utils.logger import ProctorLogger
from models.detection_models import DetectionResult, SessionData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Proctor Backend",
    description="Real-time video proctoring with AI detection",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://localhost:5175",  # Vite might use different port
        "https://video-proctoring-web-1.onrender.com",  # Your Vercel domain
        "https://*.vercel.app",  # All Vercel domains
        "https://vercel.app"     # Vercel root
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
face_detector = FaceDetector()
object_detector = ObjectDetector()
gaze_detector = GazeDetector()
proctor_logger = ProctorLogger()

# Active sessions
active_sessions: Dict[str, SessionData] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize detection models on startup"""
    logger.info("🚀 Starting AI Proctor Backend...")
    
    try:
        await face_detector.initialize()
        await object_detector.initialize()
        await gaze_detector.initialize()
        logger.info("✅ All AI models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "AI Proctor Backend",
        "status": "running",
        "models": {
            "face_detector": face_detector.is_ready,
            "object_detector": object_detector.is_ready,
            "gaze_detector": gaze_detector.is_ready
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions)
    }

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

# Frame processing optimization
frame_cache = {}
last_processing_time = {}

async def process_frame(frame: np.ndarray, session_id: str) -> DetectionResult:
    """Process a single frame through all detection models with optimization"""
    try:
        current_time = asyncio.get_event_loop().time()
        
        # Skip processing if last processing was too recent (rate limiting)
        if session_id in last_processing_time:
            if current_time - last_processing_time[session_id] < 1.0:  # Min 1 second between processing
                # Return last cached result if available
                if session_id in frame_cache:
                    return frame_cache[session_id]
        
        last_processing_time[session_id] = current_time
        
        # Resize frame for faster processing (50% reduction)
        height, width = frame.shape[:2]
        new_height, new_width = height // 2, width // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Run detections in parallel for better performance
        face_task = asyncio.create_task(face_detector.detect(resized_frame))
        object_task = asyncio.create_task(object_detector.detect(resized_frame))
        gaze_task = asyncio.create_task(gaze_detector.detect(resized_frame))
        
        # Wait for all detections to complete
        face_result = await face_task
        object_result = await object_task
        gaze_result = await gaze_task
        
        # Combine results
        result = DetectionResult(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            face_count=face_result.get('count', 0),
            face_confidence=face_result.get('confidence', 0.0),
            face_locations=face_result.get('locations', []),
            
            focus_status=gaze_result.get('focus_status', 'unknown'),
            gaze_direction=gaze_result.get('direction', 'center'),
            head_pose=gaze_result.get('head_pose', {}),
            drowsiness_score=gaze_result.get('drowsiness', 0.0),
            
            objects_detected=object_result.get('objects', []),
            suspicious_count=len([obj for obj in object_result.get('objects', []) 
                                if obj.get('class') in ['cell phone', 'book', 'laptop']]),
            
            integrity_score=calculate_integrity_score(face_result, object_result, gaze_result)
        )
        
        # Cache the result for potential reuse
        frame_cache[session_id] = result
        
        # Log significant events
        if result.face_count == 0:
            proctor_logger.log_event(session_id, "no_face_detected", {"timestamp": result.timestamp})
        elif result.face_count > 1:
            proctor_logger.log_event(session_id, "multiple_faces", {"count": result.face_count})
        
        if result.suspicious_count > 0:
            proctor_logger.log_event(session_id, "suspicious_object", {
                "objects": [obj['class'] for obj in result.objects_detected 
                           if obj.get('class') in ['cell phone', 'book', 'laptop']]
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return DetectionResult(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            error=str(e)
        )

def calculate_integrity_score(face_result: dict, object_result: dict, gaze_result: dict) -> float:
    """Calculate integrity score based on detection results"""
    score = 100.0
    
    # Face detection penalties
    face_count = face_result.get('count', 0)
    if face_count == 0:
        score -= 15
    elif face_count > 1:
        score -= 20
    
    # Gaze/focus penalties  
    if gaze_result.get('focus_status') == 'poor':
        score -= 10
    if gaze_result.get('drowsiness', 0) > 0.5:
        score -= 10
    
    # Object detection penalties
    suspicious_objects = [obj for obj in object_result.get('objects', []) 
                         if obj.get('class') in ['cell phone', 'book', 'laptop']]
    score -= len(suspicious_objects) * 25
    
    return max(0.0, score)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    
    # Create session data
    active_sessions[session_id] = SessionData(
        session_id=session_id,
        start_time=datetime.now(),
        candidate_name="",
        session_type=""
    )
    
    logger.info(f"🔌 New WebSocket connection: {session_id}")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'frame':
                # Decode and process frame
                frame = decode_base64_image(message.get('image', ''))
                
                if frame is not None:
                    # Process frame through AI models
                    result = await process_frame(frame, session_id)
                    
                    # Send results back to frontend
                    await websocket.send_text(result.model_dump_json())
                else:
                    await websocket.send_text(json.dumps({
                        "error": "Failed to decode image"
                    }))
            
            elif message.get('type') == 'session_info':
                # Update session info
                if session_id in active_sessions:
                    active_sessions[session_id].candidate_name = message.get('candidate_name', '')
                    active_sessions[session_id].session_type = message.get('session_type', '')
                    
                await websocket.send_text(json.dumps({
                    "status": "session_info_updated"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
        
        # Clean up session
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            session_data.end_time = datetime.now()
            
            # Generate final report
            await proctor_logger.generate_report(session_data)
            
            # Remove from active sessions
            del active_sessions[session_id]
    
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        await websocket.close()

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": session.session_id,
                "candidate_name": session.candidate_name,
                "start_time": session.start_time.isoformat(),
                "duration": str(datetime.now() - session.start_time)
            }
            for session in active_sessions.values()
        ]
    }

@app.get("/sessions/completed")
async def get_completed_sessions():
    """Get list of completed sessions with reports"""
    try:
        import os
        from pathlib import Path
        
        reports_dir = Path("logs/reports")
        completed_sessions = []
        
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                try:
                    # Parse session info from filename
                    filename = report_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        session_id = '_'.join(parts[:-2])  # Everything except last 2 parts (timestamp)
                        
                        # Load report to get session details
                        with open(report_file, 'r') as f:
                            import json
                            report_data = json.load(f)
                            
                        session_data = report_data.get('session_data', {})
                        
                        completed_sessions.append({
                            "session_id": session_id,
                            "candidate_name": session_data.get('candidate_name', 'Unknown'),
                            "session_type": session_data.get('session_type', 'Unknown'),
                            "start_time": session_data.get('start_time', ''),
                            "end_time": session_data.get('end_time', ''),
                            "duration_minutes": report_data.get('duration_minutes', 0),
                            "total_violations": report_data.get('total_violations', 0),
                            "risk_level": report_data.get('risk_level', 'unknown'),
                            "report_file": str(report_file)
                        })
                except Exception as e:
                    logger.warning(f"Error processing report file {report_file}: {e}")
        
        # Sort by start time (most recent first)
        completed_sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return {
            "completed_sessions": len(completed_sessions),
            "sessions": completed_sessions[:50]  # Limit to 50 most recent
        }
        
    except Exception as e:
        logger.error(f"Error getting completed sessions: {e}")
        return {
            "completed_sessions": 0,
            "sessions": []
        }

@app.get("/reports/{session_id}")
async def get_session_report(session_id: str):
    """Get detailed report for a session"""
    try:
        report = await proctor_logger.get_report(session_id)
        if report:
            return report
        else:
            raise HTTPException(status_code=404, detail="Session report not found")
    except Exception as e:
        logger.error(f"Error retrieving report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve report")

@app.get("/export/{session_id}/json")
async def export_session_json(session_id: str):
    """Export session data as JSON file"""
    try:
        report = await proctor_logger.get_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Session report not found")
        
        # Create JSON content
        import json
        
        json_content = json.dumps(report, indent=2)
        filename = f"session_report_{session_id}.json"
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to export session data")

@app.get("/export/{session_id}/csv")
async def export_session_csv(session_id: str):
    """Export session alerts as CSV file"""
    try:
        report = await proctor_logger.get_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Session report not found")
        
        import io
        import csv
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Timestamp', 'Alert Type', 'Severity', 'Message', 'Details'])
        
        # Write alert data
        for alert in report.get('alerts', []):
            writer.writerow([
                alert.get('timestamp', ''),
                alert.get('alert_type', ''),
                alert.get('severity', ''),
                alert.get('message', ''),
                str(alert.get('details', {}))
            ])
        
        csv_content = output.getvalue()
        filename = f"session_alerts_{session_id}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to export session alerts")

@app.get("/export/{session_id}/pdf")
async def export_session_pdf(session_id: str):
    """Export session report as PDF file"""
    try:
        report = await proctor_logger.get_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Session report not found")
        
        from fastapi.responses import StreamingResponse
        import io
        from datetime import datetime
        
        # Simple text-based PDF alternative (HTML that browsers can print to PDF)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Proctoring Report - {session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 20px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-left: 4px solid #ff6b6b; background: #ffe0e0; }}
                .success {{ border-left-color: #51cf66; background: #e0ffe0; }}
                .warning {{ border-left-color: #ffd43b; background: #fff8e0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Proctoring Session Report</h1>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Session Summary</h2>
                <p><strong>Candidate:</strong> {report.get('session_data', {}).get('candidate_name', 'N/A')}</p>
                <p><strong>Duration:</strong> {report.get('duration_minutes', 0):.1f} minutes</p>
                <p><strong>Total Violations:</strong> {report.get('total_violations', 0)}</p>
                <p><strong>Risk Level:</strong> {report.get('risk_level', 'Unknown').title()}</p>
            </div>
            
            <div class="section">
                <h2>Violations Breakdown</h2>
                <table>
                    <tr><th>Violation Type</th><th>Count</th></tr>
        """
        
        for violation_type, count in report.get('violation_breakdown', {}).items():
            html_content += f"<tr><td>{violation_type.replace('_', ' ').title()}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report.get('recommendations', []):
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Complete Alert Timeline</h2>
                <p><strong>Total Alerts:</strong> {len(report.get('alerts', []))}</p>
        """
        
        # Show ALL alerts, not just first 10
        for alert in report.get('alerts', []):
            severity_class = alert.get('severity', 'info')
            if severity_class == 'high': severity_class = 'alert'
            elif severity_class == 'medium': severity_class = 'warning'
            else: severity_class = 'success'
            
            # Get additional details for the alert
            details = alert.get('details', {})
            details_text = ""
            if details:
                if isinstance(details, dict):
                    details_items = []
                    for key, value in details.items():
                        if key == 'objects' and isinstance(value, list):
                            details_items.append(f"Objects: {', '.join(value)}")
                        else:
                            details_items.append(f"{key}: {value}")
                    if details_items:
                        details_text = f"<br><small>Details: {'; '.join(details_items)}</small>"
                else:
                    details_text = f"<br><small>Details: {details}</small>"
            
            html_content += f"""
                <div class="alert {severity_class}">
                    <strong>{alert.get('timestamp', '')}:</strong> {alert.get('message', '')}
                    {details_text}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Complete Session Log</h2>
                <p>All events recorded during the session:</p>
        """
        
        # Load and display ALL session events (not just alerts)
        try:
            session_events = await proctor_logger._load_session_events(session_id)
            html_content += f"<p><strong>Total Events:</strong> {len(session_events)}</p>"
            
            if session_events:
                html_content += "<table><tr><th>Timestamp</th><th>Event Type</th><th>Details</th></tr>"
                
                for event in session_events:
                    event_type = event.get('event_type', 'Unknown')
                    timestamp = event.get('timestamp', '')
                    event_data = event.get('data', {})
                    
                    # Format event data for display
                    data_str = ""
                    if isinstance(event_data, dict):
                        data_parts = []
                        for key, value in event_data.items():
                            if key == 'objects' and isinstance(value, list):
                                data_parts.append(f"Objects: {', '.join(value)}")
                            else:
                                data_parts.append(f"{key}: {value}")
                        data_str = "; ".join(data_parts)
                    else:
                        data_str = str(event_data)
                    
                    html_content += f"<tr><td>{timestamp}</td><td>{event_type}</td><td>{data_str}</td></tr>"
                
                html_content += "</table>"
            else:
                html_content += "<p>No events recorded for this session.</p>"
        except Exception as e:
            html_content += f"<p>Error loading session events: {e}</p>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        filename = f"session_report_{session_id}.html"
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to export session report")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )