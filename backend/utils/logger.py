import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import aiofiles

from models.detection_models import SessionData, AlertEvent, SessionReport

logger = logging.getLogger(__name__)

class ProctorLogger:
    """Logger for proctoring sessions, events, and reports"""
    
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "sessions").mkdir(exist_ok=True)
        (self.base_dir / "reports").mkdir(exist_ok=True)
        (self.base_dir / "events").mkdir(exist_ok=True)
        
        self.session_events: Dict[str, List[Dict]] = {}
        
    async def start_session(self, session_data: SessionData):
        """Start logging for a new session"""
        session_id = session_data.session_id
        
        # Initialize event list for session
        self.session_events[session_id] = []
        
        # Log session start
        session_start_data = {
            "session_id": session_id,
            "candidate_name": session_data.candidate_name,
            "session_type": session_data.session_type,
            "start_time": session_data.start_time.isoformat(),
            "status": "started"
        }
        
        # Save session start info
        session_file = self.base_dir / "sessions" / f"{session_id}.json"
        async with aiofiles.open(session_file, 'w') as f:
            await f.write(json.dumps(session_start_data, indent=2))
        
        logger.info(f"📝 Started logging for session {session_id}")
    
    def log_event(self, session_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log an event for a session"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "event_type": event_type,
                "data": event_data
            }
            
            # Add to memory store
            if session_id not in self.session_events:
                self.session_events[session_id] = []
            
            self.session_events[session_id].append(event)
            
            # For critical events, immediately write to disk
            if event_type in ["multiple_faces", "suspicious_object", "session_ended"]:
                self._write_event_to_disk(event)
                
        except Exception as e:
            logger.error(f"Error logging event: {e}")
    
    def _write_event_to_disk(self, event: Dict[str, Any]):
        """Write a single event to disk immediately"""
        try:
            session_id = event["session_id"]
            timestamp = datetime.now().strftime("%Y%m%d")
            
            event_file = self.base_dir / "events" / f"{session_id}_{timestamp}.jsonl"
            
            with open(event_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing event to disk: {e}")
    
    async def end_session(self, session_data: SessionData):
        """End logging for a session"""
        session_id = session_data.session_id
        
        try:
            # Update session file with end info
            session_file = self.base_dir / "sessions" / f"{session_id}.json"
            
            if session_file.exists():
                async with aiofiles.open(session_file, 'r') as f:
                    content = await f.read()
                    session_info = json.loads(content)
            else:
                session_info = {}
            
            # Update with end information
            session_info.update({
                "end_time": session_data.end_time.isoformat() if session_data.end_time else datetime.now().isoformat(),
                "total_frames_processed": session_data.total_frames_processed,
                "total_alerts": session_data.total_alerts,
                "average_integrity_score": session_data.average_integrity_score,
                "status": "completed"
            })
            
            async with aiofiles.open(session_file, 'w') as f:
                await f.write(json.dumps(session_info, indent=2))
            
            # Write all events to disk
            await self._write_session_events(session_id)
            
            logger.info(f"📝 Ended logging for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session log: {e}")
    
    async def _write_session_events(self, session_id: str):
        """Write all session events to disk"""
        try:
            if session_id not in self.session_events:
                return
            
            events = self.session_events[session_id]
            timestamp = datetime.now().strftime("%Y%m%d")
            
            event_file = self.base_dir / "events" / f"{session_id}_{timestamp}.jsonl"
            
            async with aiofiles.open(event_file, 'w') as f:
                for event in events:
                    await f.write(json.dumps(event) + '\n')
            
            # Clear from memory
            del self.session_events[session_id]
            
        except Exception as e:
            logger.error(f"Error writing session events: {e}")
    
    async def generate_report(self, session_data: SessionData) -> SessionReport:
        """Generate a comprehensive session report"""
        try:
            session_id = session_data.session_id
            
            # Load session events
            events = await self._load_session_events(session_id)
            
            # Create alert events from logged events
            alerts = self._create_alerts_from_events(events)
            
            # Calculate session statistics
            duration_minutes = 0.0
            if session_data.end_time and session_data.start_time:
                duration = session_data.end_time - session_data.start_time
                duration_minutes = duration.total_seconds() / 60
            
            # Analyze violations
            violation_breakdown = self._analyze_violations(events)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                session_data, violation_breakdown, duration_minutes
            )
            
            # Determine risk level
            risk_level = self._calculate_risk_level(
                session_data.average_integrity_score, 
                len(alerts),
                violation_breakdown
            )
            
            # Create report
            report = SessionReport(
                session_data=session_data,
                alerts=alerts,
                duration_minutes=duration_minutes,
                total_violations=len(alerts),
                violation_breakdown=violation_breakdown,
                recommendations=recommendations,
                risk_level=risk_level
            )
            
            # Save report to disk
            await self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            # Return basic report on error
            return SessionReport(
                session_data=session_data,
                recommendations=["Error occurred during report generation"],
                risk_level="unknown"
            )
    
    async def _load_session_events(self, session_id: str) -> List[Dict]:
        """Load all events for a session"""
        events = []
        
        try:
            # Check memory first
            if session_id in self.session_events:
                events.extend(self.session_events[session_id])
            
            # Load from disk
            event_files = list((self.base_dir / "events").glob(f"{session_id}_*.jsonl"))
            
            for event_file in event_files:
                async with aiofiles.open(event_file, 'r') as f:
                    content = await f.read()
                    lines = content.strip().split('\n')
                    
                    for line in lines:
                        if line.strip():
                            events.append(json.loads(line))
            
        except Exception as e:
            logger.error(f"Error loading session events: {e}")
        
        return events
    
    def _create_alerts_from_events(self, events: List[Dict]) -> List[AlertEvent]:
        """Convert events to alert objects"""
        alerts = []
        
        for event in events:
            event_type = event.get("event_type", "")
            
            if event_type in ["no_face_detected", "multiple_faces", "suspicious_object", "poor_focus", "drowsiness"]:
                severity = "high" if event_type in ["multiple_faces", "suspicious_object"] else "medium"
                
                alert = AlertEvent(
                    timestamp=event["timestamp"],
                    session_id=event["session_id"],
                    alert_type=event_type,
                    severity=severity,
                    message=self._get_alert_message(event_type, event.get("data", {})),
                    details=event.get("data", {})
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _get_alert_message(self, event_type: str, data: Dict) -> str:
        """Generate human-readable alert message"""
        messages = {
            "no_face_detected": "No face detected in frame",
            "multiple_faces": f"Multiple faces detected ({data.get('count', 'unknown')})",
            "suspicious_object": f"Suspicious object detected: {', '.join(data.get('objects', []))}",
            "poor_focus": "Candidate not looking at camera",
            "drowsiness": f"Drowsiness detected ({data.get('score', 0):.1%})"
        }
        
        return messages.get(event_type, f"Event: {event_type}")
    
    def _analyze_violations(self, events: List[Dict]) -> Dict[str, int]:
        """Analyze violation types and counts"""
        breakdown = {}
        
        for event in events:
            event_type = event.get("event_type", "")
            if event_type in ["no_face_detected", "multiple_faces", "suspicious_object", "poor_focus", "drowsiness"]:
                breakdown[event_type] = breakdown.get(event_type, 0) + 1
        
        return breakdown
    
    def _generate_recommendations(self, session_data: SessionData, violations: Dict[str, int], duration: float) -> List[str]:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        if violations.get("multiple_faces", 0) > 0:
            recommendations.append("Ensure only the candidate is visible in the camera frame")
        
        if violations.get("suspicious_object", 0) > 0:
            recommendations.append("Remove all electronic devices and unauthorized materials from the testing area")
        
        if violations.get("no_face_detected", 0) > duration * 0.1:  # More than 10% of time
            recommendations.append("Maintain consistent positioning in front of the camera")
        
        if violations.get("poor_focus", 0) > duration * 0.2:  # More than 20% of time
            recommendations.append("Keep eyes focused on the screen during the assessment")
        
        if session_data.average_integrity_score < 70:
            recommendations.append("Review session recording for detailed analysis of violations")
        
        if not recommendations:
            recommendations.append("Excellent performance with minimal violations detected")
        
        return recommendations
    
    def _calculate_risk_level(self, avg_integrity: float, alert_count: int, violations: Dict[str, int]) -> str:
        """Calculate overall risk level"""
        
        # High risk indicators
        if violations.get("multiple_faces", 0) > 3:
            return "critical"
        if violations.get("suspicious_object", 0) > 2:
            return "critical"
        
        # Medium risk indicators
        if avg_integrity < 60:
            return "high"
        if alert_count > 10:
            return "high"
        
        # Low risk
        if avg_integrity > 85 and alert_count < 5:
            return "low"
        
        return "medium"
    
    async def _save_report(self, report: SessionReport):
        """Save report to disk"""
        try:
            session_id = report.session_data.session_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report_file = self.base_dir / "reports" / f"{session_id}_{timestamp}.json"
            
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(report.model_dump_json(indent=2))
            
            logger.info(f"📊 Saved report for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    async def get_report(self, session_id: str) -> Dict:
        """Retrieve the latest report for a session"""
        try:
            report_files = list((self.base_dir / "reports").glob(f"{session_id}_*.json"))
            
            if not report_files:
                return None
            
            # Get the most recent report
            latest_report = sorted(report_files)[-1]
            
            async with aiofiles.open(latest_report, 'r') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Error retrieving report: {e}")
            return None