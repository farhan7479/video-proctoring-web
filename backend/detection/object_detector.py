import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
from typing import Dict, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Real object detection using YOLOv8 for suspicious items"""
    
    # COCO class names that are considered suspicious in exam context
    SUSPICIOUS_CLASSES = {
        'cell phone': 67,
        'book': 84, 
        'laptop': 73,
        'mouse': 74,
        'keyboard': 76,
        'remote': 75,
        'scissors': 87,
        'teddy bear': 88,  # Sometimes misclassified electronics
        'hair drier': 89,  # Sometimes misclassified phones
        'toothbrush': 90   # Sometimes misclassified stylus/pen
    }
    
    # Additional mapping for better detection
    CLASS_MAPPING = {
        67: 'cell phone',
        73: 'laptop', 
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        84: 'book',
        87: 'scissors'
    }
    
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
    async def initialize(self):
        """Initialize YOLOv8 model"""
        try:
            logger.info("📱 Initializing Object Detector (YOLOv8)...")
            
            # Load YOLOv8 model (will download if not present)
            # Using nano version for better performance
            self.model = YOLO('yolov8n.pt')  
            
            # Warm up the model with a dummy inference
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            
            self.is_ready = True
            logger.info("✅ YOLOv8 Object Detection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize object detector: {e}")
            raise
    
    async def detect(self, frame: np.ndarray) -> Dict:
        """Detect suspicious objects in frame"""
        if not self.is_ready:
            return {"objects": [], "suspicious_count": 0}
        
        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            
            detected_objects = []
            suspicious_count = 0
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        cls_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        bbox = box.xyxy.cpu().numpy()[0].astype(int)
                        
                        # Get class name
                        class_name = self.model.names.get(cls_id, 'unknown')
                        
                        # Check if it's a suspicious object
                        is_suspicious = (
                            cls_id in self.CLASS_MAPPING or 
                            class_name.lower() in [cls.lower() for cls in self.SUSPICIOUS_CLASSES.keys()]
                        )
                        
                        object_data = {
                            "class": class_name,
                            "class_id": cls_id,
                            "confidence": confidence,
                            "bbox": bbox.tolist(),  # [x1, y1, x2, y2]
                            "is_suspicious": is_suspicious
                        }
                        
                        detected_objects.append(object_data)
                        
                        if is_suspicious:
                            suspicious_count += 1
            
            return {
                "objects": detected_objects,
                "suspicious_count": suspicious_count,
                "total_objects": len(detected_objects)
            }
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return {"objects": [], "suspicious_count": 0, "error": str(e)}
    
    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw object detection bounding boxes on frame"""
        try:
            annotated_frame = frame.copy()
            
            for obj in detections.get("objects", []):
                bbox = obj["bbox"]  # [x1, y1, x2, y2]
                class_name = obj["class"]
                confidence = obj["confidence"]
                is_suspicious = obj.get("is_suspicious", False)
                
                x1, y1, x2, y2 = bbox
                
                # Color coding: red for suspicious, green for normal
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                thickness = 3 if is_suspicious else 2
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with background
                label = f"{class_name}: {confidence:.2f}"
                if is_suspicious:
                    label = f"⚠️ {label}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background rectangle
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), 
                             color, -1)
                
                # Draw text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add summary info
            suspicious_count = detections.get("suspicious_count", 0)
            if suspicious_count > 0:
                summary = f"⚠️ {suspicious_count} suspicious object(s) detected"
                cv2.putText(annotated_frame, summary, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing object detections: {e}")
            return frame
    
    def get_suspicious_objects(self, detections: Dict) -> List[Dict]:
        """Get only suspicious objects from detections"""
        return [obj for obj in detections.get("objects", []) 
                if obj.get("is_suspicious", False)]
    
    def update_confidence_threshold(self, threshold: float):
        """Update detection confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def add_custom_suspicious_class(self, class_name: str, class_id: int):
        """Add a custom class as suspicious"""
        self.SUSPICIOUS_CLASSES[class_name.lower()] = class_id
        self.CLASS_MAPPING[class_id] = class_name.lower()
        logger.info(f"Added {class_name} (ID: {class_id}) as suspicious class")