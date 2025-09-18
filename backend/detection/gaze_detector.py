import cv2
import numpy as np
import asyncio
from typing import Dict, List, Tuple
import logging
import math

# Try to import MediaPipe, fallback to None if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)

class GazeDetector:
    """Gaze and focus detection using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.is_ready = False
        
        # Eye aspect ratio for drowsiness detection
        self.ear_threshold = 0.2
        self.drowsy_frames = 0
        self.drowsy_threshold = 20  # frames
        
        # Gaze direction thresholds
        self.gaze_threshold = 0.15
        
    async def initialize(self):
        """Initialize MediaPipe Face Mesh for gaze detection or fallback mode"""
        try:
            logger.info("👁️ Initializing Gaze Detector...")
            
            if MEDIAPIPE_AVAILABLE:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Face Mesh initialized for gaze detection")
            else:
                logger.warning("⚠️ MediaPipe not available, using basic gaze estimation")
            
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize gaze detector: {e}")
            # Still set as ready for basic functionality
            self.is_ready = True
            logger.info("✅ Gaze detector initialized in fallback mode")
    
    async def detect(self, frame: np.ndarray) -> Dict:
        """Detect gaze direction and focus status"""
        if not self.is_ready:
            return {
                "focus_status": "unknown",
                "direction": "center", 
                "head_pose": {},
                "drowsiness": 0.0
            }
        
        try:
            if MEDIAPIPE_AVAILABLE and self.face_mesh:
                # Use MediaPipe for detailed gaze detection
                return await self._detect_with_mediapipe(frame)
            else:
                # Use basic fallback gaze estimation
                return await self._detect_fallback(frame)
                
        except Exception as e:
            logger.error(f"Gaze detection error: {e}")
            return {
                "focus_status": "error",
                "direction": "unknown",
                "head_pose": {},
                "drowsiness": 0.0,
                "error": str(e)
            }
    
    async def _detect_with_mediapipe(self, frame: np.ndarray) -> Dict:
        """MediaPipe-based gaze detection"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate gaze direction
            gaze_info = self._calculate_gaze_direction(face_landmarks, frame.shape)
            
            # Calculate head pose
            head_pose = self._calculate_head_pose(face_landmarks, frame.shape)
            
            # Calculate eye aspect ratio for drowsiness
            ear = self._calculate_eye_aspect_ratio(face_landmarks)
            drowsiness = self._update_drowsiness_score(ear)
            
            # Determine focus status
            focus_status = self._determine_focus_status(gaze_info, head_pose)
            
            return {
                "focus_status": focus_status,
                "direction": gaze_info["direction"],
                "gaze_vector": gaze_info.get("vector", [0, 0]),
                "head_pose": head_pose,
                "drowsiness": drowsiness,
                "eye_aspect_ratio": ear
            }
        else:
            # No face detected
            return {
                "focus_status": "no_face",
                "direction": "unknown",
                "head_pose": {},
                "drowsiness": 0.0
            }
    
    async def _detect_fallback(self, frame: np.ndarray) -> Dict:
        """Fallback gaze detection using basic computer vision"""
        # Very basic gaze estimation - assumes person is looking at camera
        # This is a simplified version for demo purposes
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade (same as face detector)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Assume decent focus if face is detected and centered
            (x, y, w, h) = faces[0]
            frame_center_x = frame.shape[1] // 2
            face_center_x = x + w // 2
            
            # Simple focus estimation based on face position
            offset = abs(face_center_x - frame_center_x) / frame_center_x
            
            if offset < 0.2:
                focus_status = "good"
                direction = "center"
            elif offset < 0.4:
                focus_status = "acceptable"
                direction = "left" if face_center_x < frame_center_x else "right"
            else:
                focus_status = "poor"
                direction = "left" if face_center_x < frame_center_x else "right"
            
            return {
                "focus_status": focus_status,
                "direction": direction,
                "head_pose": {"yaw": offset * 30, "pitch": 0, "roll": 0},  # Rough estimate
                "drowsiness": 0.0  # Not available in fallback mode
            }
        else:
            return {
                "focus_status": "no_face",
                "direction": "unknown",
                "head_pose": {},
                "drowsiness": 0.0
            }
    
    def _calculate_gaze_direction(self, face_landmarks, frame_shape) -> Dict:
        """Calculate gaze direction from face landmarks"""
        try:
            height, width = frame_shape[:2]
            
            # Get key eye landmarks
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Get eye centers
            left_eye_center = self._get_landmark_center(face_landmarks, left_eye_landmarks, width, height)
            right_eye_center = self._get_landmark_center(face_landmarks, right_eye_landmarks, width, height)
            
            # Get nose tip (landmark 1)
            nose_tip = face_landmarks.landmark[1]
            nose_point = [nose_tip.x * width, nose_tip.y * height]
            
            # Calculate eye line vector
            eye_vector = [
                right_eye_center[0] - left_eye_center[0],
                right_eye_center[1] - left_eye_center[1]
            ]
            
            # Calculate gaze vector (simplified)
            eye_center = [
                (left_eye_center[0] + right_eye_center[0]) / 2,
                (left_eye_center[1] + right_eye_center[1]) / 2
            ]
            
            gaze_x = (eye_center[0] - width/2) / (width/2)
            gaze_y = (eye_center[1] - height/2) / (height/2)
            
            # Determine direction
            direction = "center"
            if abs(gaze_x) > self.gaze_threshold:
                direction = "right" if gaze_x > 0 else "left"
            if abs(gaze_y) > self.gaze_threshold:
                direction += "_up" if gaze_y < 0 else "_down"
            
            return {
                "direction": direction,
                "vector": [gaze_x, gaze_y],
                "eye_centers": {"left": left_eye_center, "right": right_eye_center}
            }
            
        except Exception as e:
            logger.error(f"Gaze calculation error: {e}")
            return {"direction": "unknown", "vector": [0, 0]}
    
    def _calculate_head_pose(self, face_landmarks, frame_shape) -> Dict:
        """Calculate head pose angles"""
        try:
            height, width = frame_shape[:2]
            
            # Get key facial landmarks for head pose
            nose_tip = face_landmarks.landmark[1]  
            chin = face_landmarks.landmark[175]
            left_eye_corner = face_landmarks.landmark[33]
            right_eye_corner = face_landmarks.landmark[362]
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            
            # Convert to pixel coordinates
            landmarks_2d = np.array([
                [nose_tip.x * width, nose_tip.y * height],
                [chin.x * width, chin.y * height],
                [left_eye_corner.x * width, left_eye_corner.y * height],
                [right_eye_corner.x * width, right_eye_corner.y * height],
                [left_mouth.x * width, left_mouth.y * height],
                [right_mouth.x * width, right_mouth.y * height]
            ], dtype=np.float64)
            
            # 3D model points
            model_points = np.array([
                [0.0, 0.0, 0.0],        # Nose tip
                [0.0, -330.0, -65.0],   # Chin
                [-225.0, 170.0, -135.0], # Left eye corner
                [225.0, 170.0, -135.0],  # Right eye corner
                [-150.0, -150.0, -125.0], # Left mouth corner
                [150.0, -150.0, -125.0]   # Right mouth corner
            ])
            
            # Camera matrix (simplified)
            camera_matrix = np.array([
                [width, 0, width/2],
                [0, width, height/2],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, landmarks_2d, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                return {
                    "pitch": float(angles[0]),  # Up/down
                    "yaw": float(angles[1]),    # Left/right  
                    "roll": float(angles[2])    # Tilt
                }
            else:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
                
        except Exception as e:
            logger.error(f"Head pose calculation error: {e}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    
    def _calculate_eye_aspect_ratio(self, face_landmarks) -> float:
        """Calculate Eye Aspect Ratio for drowsiness detection"""
        try:
            # Left eye landmarks
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Right eye landmarks  
            right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate EAR for both eyes
            left_ear = self._eye_aspect_ratio_for_eye(face_landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._eye_aspect_ratio_for_eye(face_landmarks, [362, 385, 387, 263, 373, 380])
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            return ear
            
        except Exception as e:
            logger.error(f"EAR calculation error: {e}")
            return 0.3  # Default value
    
    def _eye_aspect_ratio_for_eye(self, face_landmarks, eye_points) -> float:
        """Calculate EAR for a single eye"""
        # Get eye landmarks
        points = []
        for point_idx in eye_points:
            landmark = face_landmarks.landmark[point_idx]
            points.append([landmark.x, landmark.y])
        
        points = np.array(points)
        
        # Calculate distances
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _update_drowsiness_score(self, ear: float) -> float:
        """Update drowsiness score based on EAR"""
        if ear < self.ear_threshold:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = max(0, self.drowsy_frames - 2)
        
        # Calculate drowsiness score (0 to 1)
        drowsiness_score = min(1.0, self.drowsy_frames / self.drowsy_threshold)
        return drowsiness_score
    
    def _determine_focus_status(self, gaze_info: Dict, head_pose: Dict) -> str:
        """Determine if person is focused based on gaze and head pose"""
        try:
            gaze_direction = gaze_info.get("direction", "unknown")
            head_yaw = abs(head_pose.get("yaw", 0))
            head_pitch = abs(head_pose.get("pitch", 0))
            
            # Thresholds for focus determination
            head_yaw_threshold = 30  # degrees
            head_pitch_threshold = 20  # degrees
            
            # Check if looking roughly at camera
            is_centered_gaze = gaze_direction in ["center", "center_up", "center_down"]
            is_head_forward = head_yaw < head_yaw_threshold and head_pitch < head_pitch_threshold
            
            if is_centered_gaze and is_head_forward:
                return "good"
            elif is_head_forward:
                return "acceptable"  
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Focus status error: {e}")
            return "unknown"
    
    def _get_landmark_center(self, face_landmarks, landmark_indices: List[int], width: int, height: int) -> List[float]:
        """Get center point of multiple landmarks"""
        x_coords = []
        y_coords = []
        
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x_coords.append(landmark.x * width)
            y_coords.append(landmark.y * height)
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return [center_x, center_y]
    
    def _rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])
    
    def draw_gaze_info(self, frame: np.ndarray, gaze_data: Dict) -> np.ndarray:
        """Draw gaze information on frame"""
        try:
            annotated_frame = frame.copy()
            
            # Draw focus status
            focus_status = gaze_data.get("focus_status", "unknown")
            color = (0, 255, 0) if focus_status == "good" else (0, 165, 255) if focus_status == "acceptable" else (0, 0, 255)
            
            cv2.putText(annotated_frame, f"Focus: {focus_status.title()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw gaze direction
            direction = gaze_data.get("direction", "unknown")
            cv2.putText(annotated_frame, f"Gaze: {direction}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw drowsiness score
            drowsiness = gaze_data.get("drowsiness", 0.0)
            drowsy_color = (0, 255, 0) if drowsiness < 0.3 else (0, 165, 255) if drowsiness < 0.6 else (0, 0, 255)
            cv2.putText(annotated_frame, f"Drowsiness: {drowsiness:.2f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsy_color, 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing gaze info: {e}")
            return frame