import cv2
import numpy as np
import asyncio
from typing import Dict, List, Tuple
import logging

# Try to import MediaPipe, fallback to None if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceDetector:
    """Real face detection using OpenCV Haar Cascades and MediaPipe"""
    
    def __init__(self):
        self.face_cascade = None
        self.mp_face_detection = None
        self.mp_drawing = None
        self.face_detection = None
        self.is_ready = False
        self.use_mediapipe = True
        
    async def initialize(self):
        """Initialize face detection models"""
        try:
            logger.info("🔍 Initializing Face Detector...")
            
            # Try MediaPipe first (more accurate) if available
            if MEDIAPIPE_AVAILABLE:
                try:
                    self.mp_face_detection = mp.solutions.face_detection
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.face_detection = self.mp_face_detection.FaceDetection(
                        model_selection=0, 
                        min_detection_confidence=0.5
                    )
                    self.use_mediapipe = True
                    logger.info("✅ MediaPipe Face Detection initialized")
                except Exception as mp_error:
                    logger.warning(f"MediaPipe failed: {mp_error}, falling back to OpenCV")
                    self.use_mediapipe = False
            else:
                logger.info("⚠️ MediaPipe not available, using OpenCV")
                self.use_mediapipe = False
            
            # Fallback to OpenCV Haar Cascades
            if not self.use_mediapipe:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if self.face_cascade.empty():
                    raise Exception("Failed to load Haar cascade classifier")
                
                logger.info("✅ OpenCV Haar Cascade initialized")
            
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize face detector: {e}")
            raise
    
    async def detect(self, frame: np.ndarray) -> Dict:
        """Detect faces in frame"""
        if not self.is_ready:
            return {"count": 0, "confidence": 0.0, "locations": []}
        
        try:
            if self.use_mediapipe:
                return await self._detect_mediapipe(frame)
            else:
                return await self._detect_opencv(frame)
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {"count": 0, "confidence": 0.0, "locations": [], "error": str(e)}
    
    async def _detect_mediapipe(self, frame: np.ndarray) -> Dict:
        """MediaPipe face detection"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.face_detection.process(rgb_frame)
            
            face_locations = []
            confidences = []
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_locations.append([x, y, width, height])
                    confidences.append(detection.score[0])
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "count": len(face_locations),
                "confidence": float(avg_confidence),
                "locations": face_locations
            }
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return {"count": 0, "confidence": 0.0, "locations": []}
    
    async def _detect_opencv(self, frame: np.ndarray) -> Dict:
        """OpenCV Haar Cascade face detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list format
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append([int(x), int(y), int(w), int(h)])
            
            # OpenCV doesn't provide confidence directly, estimate based on detection
            confidence = 0.8 if len(faces) > 0 else 0.0
            
            return {
                "count": len(face_locations),
                "confidence": confidence,
                "locations": face_locations
            }
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return {"count": 0, "confidence": 0.0, "locations": []}
    
    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw face detection bounding boxes on frame"""
        try:
            annotated_frame = frame.copy()
            
            for location in detections.get("locations", []):
                x, y, w, h = location
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                label = f"Face ({detections.get('confidence', 0):.2f})"
                cv2.putText(annotated_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing face detections: {e}")
            return frame