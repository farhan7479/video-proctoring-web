from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class DetectionResult(BaseModel):
    """Result of AI detection processing"""
    timestamp: str
    session_id: str
    
    # Face detection
    face_count: int = 0
    face_confidence: float = 0.0
    face_locations: List[List[int]] = []
    
    # Gaze/Focus detection
    focus_status: str = "unknown"  # good, acceptable, poor, no_face
    gaze_direction: str = "center"
    head_pose: Dict[str, float] = {}
    drowsiness_score: float = 0.0
    
    # Object detection
    objects_detected: List[Dict[str, Any]] = []
    suspicious_count: int = 0
    
    # Overall metrics
    integrity_score: float = 100.0
    
    # Error handling
    error: Optional[str] = None

class SessionData(BaseModel):
    """Session information and metadata"""
    session_id: str
    candidate_name: str = ""
    session_type: str = ""  # technical, behavioral, assessment, exam
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Session statistics
    total_frames_processed: int = 0
    total_alerts: int = 0
    average_integrity_score: float = 100.0
    
    # Detection summaries
    face_detection_summary: Dict[str, Any] = {}
    object_detection_summary: Dict[str, Any] = {}
    focus_detection_summary: Dict[str, Any] = {}
    
class AlertEvent(BaseModel):
    """Individual alert/violation event"""
    timestamp: str
    session_id: str
    alert_type: str  # no_face, multiple_faces, suspicious_object, poor_focus, drowsiness
    severity: str  # low, medium, high, critical
    message: str
    details: Dict[str, Any] = {}
    frame_number: Optional[int] = None
    
class SessionReport(BaseModel):
    """Complete session report"""
    session_data: SessionData
    alerts: List[AlertEvent] = []
    detection_timeline: List[DetectionResult] = []
    
    # Summary statistics
    duration_minutes: float = 0.0
    total_violations: int = 0
    violation_breakdown: Dict[str, int] = {}
    integrity_trend: List[float] = []
    
    # Recommendations
    recommendations: List[str] = []
    risk_level: str = "low"  # low, medium, high, critical

class ProctorSettings(BaseModel):
    """Configuration settings for proctoring"""
    
    # Face detection settings
    face_detection_confidence: float = 0.5
    face_timeout_seconds: int = 10
    
    # Gaze detection settings
    gaze_sensitivity: float = 0.15
    focus_timeout_seconds: int = 5
    drowsiness_threshold: float = 0.5
    
    # Object detection settings
    object_detection_confidence: float = 0.5
    suspicious_classes: List[str] = [
        'cell phone', 'book', 'laptop', 'mouse', 'keyboard'
    ]
    
    # Alert settings
    enable_audio_alerts: bool = True
    alert_cooldown_seconds: int = 3
    
    # Recording settings
    save_annotated_frames: bool = False
    frame_save_interval: int = 30  # seconds

class SystemStatus(BaseModel):
    """System health and status information"""
    timestamp: str
    
    # Model status
    face_detector_ready: bool = False
    object_detector_ready: bool = False
    gaze_detector_ready: bool = False
    
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    frames_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Active sessions
    active_sessions_count: int = 0
    total_frames_processed: int = 0
    
    # Errors
    recent_errors: List[str] = []