import './style.css'

// Application State
const AppState = {
  isSessionActive: false,
  sessionStartTime: null,
  sessionId: null,
  videoStream: null,
  detectionInterval: null,
  sessionTimer: null,
  alerts: [],
  currentCandidate: '',
  sessionData: {
    faceCount: 0,
    focusStatus: 'Good',
    drowsinessScore: 0.0,
    objectCount: 0,
    integrityScore: 100
  }
}

// DOM Elements
const elements = {
  sessionForm: document.getElementById('sessionForm'),
  candidateName: document.getElementById('candidateName'),
  sessionType: document.getElementById('sessionType'),
  setupSection: document.getElementById('setupSection'),
  dashboardSection: document.getElementById('dashboardSection'),
  videoElement: document.getElementById('videoElement'),
  canvasElement: document.getElementById('canvasElement'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  toastContainer: document.getElementById('toastContainer'),
  
  // Dashboard elements
  sessionTitle: document.getElementById('sessionTitle'),
  candidateDisplay: document.getElementById('candidateDisplay'),
  sessionTimer: document.getElementById('sessionTimer'),
  pauseSession: document.getElementById('pauseSession'),
  stopSession: document.getElementById('stopSession'),
  
  // Metrics
  faceCount: document.getElementById('faceCount'),
  focusStatus: document.getElementById('focusStatus'),
  drowsinessScore: document.getElementById('drowsinessScore'),
  objectCount: document.getElementById('objectCount'),
  integrityScore: document.getElementById('integrityScore'),
  integrityBadge: document.getElementById('integrityBadge'),
  integrityDetails: document.getElementById('integrityDetails'),
  
  // Status indicators
  faceStatus: document.getElementById('faceStatus'),
  focusIndicator: document.getElementById('focusIndicator'),
  drowsinessStatus: document.getElementById('drowsinessStatus'),
  objectStatus: document.getElementById('objectStatus'),
  
  // Alerts
  alertCounter: document.getElementById('alertCounter'),
  alertsList: document.getElementById('alertsList'),
  
  // Settings
  focusTimeout: document.getElementById('focusTimeout'),
  faceTimeout: document.getElementById('faceTimeout'),
  confidenceThreshold: document.getElementById('confidenceThreshold'),
  
  // FPS counter
  fpsCounter: document.getElementById('fpsCounter')
}

// Utility Functions
const utils = {
  formatTime: (seconds) => {
    const hrs = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  },

  showToast: (title, message, type = 'info') => {
    const toast = document.createElement('div')
    toast.className = `bg-white rounded-lg shadow-xl border border-gray-200 p-4 min-w-[300px] transform translate-x-full opacity-0 transition-all duration-300`
    
    const typeColors = {
      success: 'text-green-600 bg-green-100',
      error: 'text-red-600 bg-red-100',
      warning: 'text-yellow-600 bg-yellow-100',
      info: 'text-blue-600 bg-blue-100'
    }
    
    const typeIcons = {
      success: 'fas fa-check-circle',
      error: 'fas fa-exclamation-circle', 
      warning: 'fas fa-exclamation-triangle',
      info: 'fas fa-info-circle'
    }

    toast.innerHTML = `
      <div class="flex items-start space-x-3">
        <div class="w-6 h-6 rounded-full flex items-center justify-center ${typeColors[type]}">
          <i class="${typeIcons[type]} text-xs"></i>
        </div>
        <div class="flex-1 min-w-0">
          <div class="font-semibold text-gray-800 text-sm">${title}</div>
          <div class="text-gray-600 text-sm mt-1">${message}</div>
        </div>
        <button class="text-gray-400 hover:text-gray-600 transition-colors" onclick="this.closest('div').remove()">
          <i class="fas fa-times text-xs"></i>
        </button>
      </div>
    `

    elements.toastContainer.appendChild(toast)
    
    // Animate in
    setTimeout(() => {
      toast.classList.remove('translate-x-full', 'opacity-0')
    }, 10)

    // Auto remove
    setTimeout(() => {
      toast.classList.add('translate-x-full', 'opacity-0')
      setTimeout(() => toast.remove(), 300)
    }, 5000)
  },

  showLoading: (show = true) => {
    if (show) {
      elements.loadingOverlay.classList.remove('hidden')
    } else {
      elements.loadingOverlay.classList.add('hidden')
    }
  },

  updateStatusIndicator: (element, status) => {
    const colors = {
      good: 'bg-green-500',
      warning: 'bg-yellow-500',
      danger: 'bg-red-500'
    }
    
    element.className = `w-3 h-3 rounded-full ${colors[status] || colors.good}`
  }
}

// WebSocket connection for real-time AI processing
const websocket = {
  connection: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 5,
  reconnectInterval: 3000,

  connect(sessionId) {
    try {
      this.connection = new WebSocket(`ws://localhost:8000/ws/${sessionId}`)
      
      this.connection.onopen = () => {
        console.log('🔌 WebSocket connected to backend')
        utils.showToast('Backend Connected', 'Real-time AI processing active', 'success')
        this.reconnectAttempts = 0
      }
      
      this.connection.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data)
          if (result.error) {
            console.error('Backend error:', result.error)
          } else {
            videoProcessing.handleDetectionResult(result)
          }
        } catch (e) {
          console.error('Error parsing detection result:', e)
        }
      }
      
      this.connection.onclose = () => {
        console.log('🔌 WebSocket disconnected')
        if (AppState.isSessionActive && this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => {
            console.log(`Reconnecting... (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`)
            this.reconnectAttempts++
            this.connect(sessionId)
          }, this.reconnectInterval)
        }
      }
      
      this.connection.onerror = (error) => {
        console.error('WebSocket error:', error)
        utils.showToast('Connection Error', 'Failed to connect to AI backend', 'error')
      }
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      utils.showToast('Connection Failed', 'Could not connect to backend', 'error')
    }
  },

  sendFrame(imageData) {
    if (this.connection && this.connection.readyState === WebSocket.OPEN) {
      const message = {
        type: 'frame',
        image: imageData
      }
      this.connection.send(JSON.stringify(message))
    }
  },

  sendSessionInfo(candidateName, sessionType) {
    if (this.connection && this.connection.readyState === WebSocket.OPEN) {
      const message = {
        type: 'session_info',
        candidate_name: candidateName,
        session_type: sessionType
      }
      this.connection.send(JSON.stringify(message))
    }
  },

  disconnect() {
    if (this.connection) {
      this.connection.close()
      this.connection = null
    }
  }
}

// Video Processing Functions
const videoProcessing = {
  lastFrameTime: 0,
  frameInterval: 1000, // Send frame every 1 second for processing

  async initializeCamera() {
    try {
      utils.showLoading(true)
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      })
      
      elements.videoElement.srcObject = stream
      AppState.videoStream = stream
      
      utils.showLoading(false)
      utils.showToast('Camera Ready', 'Video feed initialized successfully', 'success')
      
      return true
    } catch (error) {
      utils.showLoading(false)
      utils.showToast('Camera Error', 'Failed to access camera: ' + error.message, 'error')
      return false
    }
  },

  stopCamera() {
    if (AppState.videoStream) {
      AppState.videoStream.getTracks().forEach(track => track.stop())
      AppState.videoStream = null
      elements.videoElement.srcObject = null
    }
  },

  startDetection() {
    // Start real-time frame capture and processing
    AppState.detectionInterval = setInterval(() => {
      this.captureAndProcessFrame()
    }, this.frameInterval)
  },

  stopDetection() {
    if (AppState.detectionInterval) {
      clearInterval(AppState.detectionInterval)
      AppState.detectionInterval = null
    }
  },

  captureAndProcessFrame() {
    try {
      if (!elements.videoElement.videoWidth || !elements.videoElement.videoHeight) {
        return
      }

      // Create canvas to capture frame
      const canvas = elements.canvasElement
      const ctx = canvas.getContext('2d')
      
      // Set canvas size to match video
      canvas.width = elements.videoElement.videoWidth
      canvas.height = elements.videoElement.videoHeight
      
      // Draw current video frame to canvas
      ctx.drawImage(elements.videoElement, 0, 0, canvas.width, canvas.height)
      
      // Convert to base64 image data
      const imageData = canvas.toDataURL('image/jpeg', 0.8)
      
      // Send to backend for AI processing
      websocket.sendFrame(imageData)
      
    } catch (error) {
      console.error('Error capturing frame:', error)
    }
  },

  handleDetectionResult(result) {
    try {
      // Convert backend result to frontend format
      const frontendData = {
        faceCount: result.face_count || 0,
        focusStatus: this.mapFocusStatus(result.focus_status),
        drowsinessScore: result.drowsiness_score || 0.0,
        objectCount: result.suspicious_count || 0,
        fps: 30 // Placeholder, backend doesn't send FPS
      }
      
      this.updateMetrics(frontendData)
      this.checkForAlerts(frontendData, result)
      
    } catch (error) {
      console.error('Error handling detection result:', error)
    }
  },

  mapFocusStatus(backendStatus) {
    const statusMap = {
      'good': 'Good',
      'acceptable': 'Good', 
      'poor': 'Poor',
      'no_face': 'Poor',
      'unknown': 'Poor'
    }
    return statusMap[backendStatus] || 'Poor'
  },

  updateMetrics(data) {
    // Update face count
    elements.faceCount.textContent = data.faceCount
    utils.updateStatusIndicator(elements.faceStatus, data.faceCount === 1 ? 'good' : 'warning')
    
    // Update focus status  
    elements.focusStatus.textContent = data.focusStatus
    utils.updateStatusIndicator(elements.focusIndicator, data.focusStatus === 'Good' ? 'good' : 'danger')
    
    // Update drowsiness
    elements.drowsinessScore.textContent = data.drowsinessScore.toFixed(1)
    utils.updateStatusIndicator(elements.drowsinessStatus, data.drowsinessScore < 0.3 ? 'good' : 'warning')
    
    // Update object count
    elements.objectCount.textContent = data.objectCount
    utils.updateStatusIndicator(elements.objectStatus, data.objectCount === 0 ? 'good' : 'danger')
    
    // Update FPS
    elements.fpsCounter.textContent = `${data.fps} FPS`
    
    // Update integrity score
    this.calculateIntegrityScore()
    
    // Store current data
    AppState.sessionData = { ...AppState.sessionData, ...data }
  },

  calculateIntegrityScore() {
    let score = 100
    
    // Deduct points based on violations
    if (AppState.sessionData.faceCount === 0) score -= 10
    if (AppState.sessionData.faceCount > 1) score -= 15
    if (AppState.sessionData.focusStatus === 'Poor') score -= 5
    if (AppState.sessionData.drowsinessScore > 0.5) score -= 5
    if (AppState.sessionData.objectCount > 0) score -= 20
    
    score = Math.max(0, score)
    AppState.sessionData.integrityScore = score
    
    // Update UI
    elements.integrityScore.textContent = score
    
    // Update badge and details
    if (score >= 90) {
      elements.integrityBadge.className = 'inline-block px-4 py-1 rounded-full text-sm font-semibold bg-green-100 text-green-800'
      elements.integrityBadge.textContent = 'Excellent'
      elements.integrityDetails.textContent = 'No violations detected'
    } else if (score >= 80) {
      elements.integrityBadge.className = 'inline-block px-4 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-800'
      elements.integrityBadge.textContent = 'Good'
      elements.integrityDetails.textContent = 'Minor violations detected'
    } else if (score >= 70) {
      elements.integrityBadge.className = 'inline-block px-4 py-1 rounded-full text-sm font-semibold bg-yellow-100 text-yellow-800'
      elements.integrityBadge.textContent = 'Acceptable'
      elements.integrityDetails.textContent = 'Some violations detected'
    } else {
      elements.integrityBadge.className = 'inline-block px-4 py-1 rounded-full text-sm font-semibold bg-red-100 text-red-800'
      elements.integrityBadge.textContent = 'Poor'
      elements.integrityDetails.textContent = 'Multiple violations detected'
    }
  },

  checkForAlerts(data, backendResult = null) {
    const now = new Date()
    
    // Check for face detection issues
    if (data.faceCount === 0) {
      this.addAlert('warning', 'No face detected', now)
    } else if (data.faceCount > 1) {
      this.addAlert('error', `Multiple faces detected: ${data.faceCount}`, now)
    }
    
    // Check for focus issues
    if (data.focusStatus === 'Poor') {
      this.addAlert('warning', 'Candidate not looking at camera', now)
    }
    
    // Check for drowsiness
    if (data.drowsinessScore > 0.5) {
      this.addAlert('warning', `Drowsiness detected (${(data.drowsinessScore * 100).toFixed(0)}%)`, now)
    }
    
    // Check for suspicious objects with detailed info from backend
    if (data.objectCount > 0 && backendResult && backendResult.objects_detected) {
      const suspiciousObjects = backendResult.objects_detected
        .filter(obj => obj.is_suspicious)
        .map(obj => obj.class)
        .join(', ')
      
      this.addAlert('error', `Suspicious objects detected: ${suspiciousObjects}`, now)
    } else if (data.objectCount > 0) {
      this.addAlert('error', `Suspicious object detected`, now)
    }
  },

  addAlert(type, message, timestamp) {
    const alert = {
      type,
      message,
      timestamp: timestamp.toLocaleTimeString(),
      id: Date.now()
    }
    
    AppState.alerts.unshift(alert)
    AppState.alerts = AppState.alerts.slice(0, 10) // Keep only last 10
    
    this.updateAlertsUI()
    
    // Show toast for high priority alerts
    if (type === 'error') {
      utils.showToast('Security Alert', message, 'error')
    }
  },

  updateAlertsUI() {
    elements.alertCounter.textContent = AppState.alerts.length
    
    if (AppState.alerts.length === 0) {
      elements.alertsList.innerHTML = `
        <div class="flex flex-col items-center justify-center py-8 text-gray-500">
          <i class="fas fa-shield-check text-3xl text-green-500 mb-2"></i>
          <span class="text-sm">All systems normal</span>
        </div>
      `
    } else {
      elements.alertsList.innerHTML = AppState.alerts.map(alert => {
        const typeColors = {
          error: 'border-red-500 bg-red-50',
          warning: 'border-yellow-500 bg-yellow-50',
          info: 'border-blue-500 bg-blue-50'
        }
        
        const typeIcons = {
          error: 'fas fa-exclamation-circle text-red-500',
          warning: 'fas fa-exclamation-triangle text-yellow-500', 
          info: 'fas fa-info-circle text-blue-500'
        }
        
        return `
          <div class="flex items-start space-x-3 p-3 rounded-lg border-l-4 ${typeColors[alert.type]} mb-3">
            <i class="${typeIcons[alert.type]} text-sm mt-0.5"></i>
            <div class="flex-1 min-w-0">
              <div class="text-xs text-gray-500 font-semibold mb-1">${alert.timestamp}</div>
              <div class="text-sm text-gray-700">${alert.message}</div>
            </div>
          </div>
        `
      }).join('')
    }
  }
}

// Session Management
const sessionManager = {
  startSession(candidateName, sessionType) {
    AppState.isSessionActive = true
    AppState.sessionStartTime = new Date()
    AppState.currentCandidate = candidateName
    AppState.alerts = []
    
    // Generate unique session ID
    const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
    AppState.sessionId = sessionId
    
    // Update UI
    elements.candidateDisplay.textContent = candidateName
    elements.setupSection.classList.add('hidden')
    elements.dashboardSection.classList.remove('hidden')
    
    // Start timer
    this.startTimer()
    
    // Connect to backend WebSocket
    websocket.connect(sessionId)
    
    // Initialize video and detection
    videoProcessing.initializeCamera().then(success => {
      if (success) {
        // Send session info to backend
        setTimeout(() => {
          websocket.sendSessionInfo(candidateName, sessionType)
        }, 1000) // Wait for WebSocket to be fully connected
        
        videoProcessing.startDetection()
        utils.showToast('Session Started', `Real-time AI monitoring for ${candidateName}`, 'success')
      }
    })
  },

  pauseSession() {
    if (AppState.sessionTimer) {
      clearInterval(AppState.sessionTimer)
      AppState.sessionTimer = null
    }
    videoProcessing.stopDetection()
    utils.showToast('Session Paused', 'Monitoring temporarily stopped', 'warning')
  },

  stopSession() {
    AppState.isSessionActive = false
    
    // Stop all monitoring
    this.stopTimer()
    videoProcessing.stopDetection()
    videoProcessing.stopCamera()
    
    // Disconnect from backend
    websocket.disconnect()
    
    // Show setup screen
    elements.dashboardSection.classList.add('hidden')
    elements.setupSection.classList.remove('hidden')
    
    // Generate report
    this.generateSessionReport()
    
    utils.showToast('Session Ended', 'AI analysis complete, report generated', 'success')
  },

  startTimer() {
    let seconds = 0
    AppState.sessionTimer = setInterval(() => {
      seconds++
      elements.sessionTimer.textContent = utils.formatTime(seconds)
    }, 1000)
  },

  stopTimer() {
    if (AppState.sessionTimer) {
      clearInterval(AppState.sessionTimer)
      AppState.sessionTimer = null
    }
  },

  generateSessionReport() {
    const endTime = new Date()
    const duration = Math.floor((endTime - AppState.sessionStartTime) / 1000)
    
    const report = {
      candidate: AppState.currentCandidate,
      startTime: AppState.sessionStartTime.toISOString(),
      endTime: endTime.toISOString(),
      duration: utils.formatTime(duration),
      integrityScore: AppState.sessionData.integrityScore,
      alerts: AppState.alerts,
      summary: AppState.sessionData
    }
    
    console.log('Session Report:', report)
    
    // Show export options after session ends
    this.showExportOptions()
  },
  
  showExportOptions() {
    const sessionId = AppState.sessionId
    
    // Update export section in UI
    const exportSection = document.getElementById('exportOptions')
    if (exportSection) {
      exportSection.innerHTML = `
        <div class="space-y-4">
          <h4 class="font-semibold text-gray-800 mb-3">Download Session Report</h4>
          <div class="grid gap-3">
            <button data-download="${sessionId}" data-format="json" 
                    class="download-btn flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors cursor-pointer">
              <div class="flex items-center space-x-3">
                <i class="fas fa-file-code text-blue-600"></i>
                <div class="text-left">
                  <div class="font-medium text-gray-800">JSON Report</div>
                  <div class="text-xs text-gray-500">Complete session data</div>
                </div>
              </div>
              <i class="fas fa-download text-blue-600"></i>
            </button>
            
            <button data-download="${sessionId}" data-format="csv" 
                    class="download-btn flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors cursor-pointer">
              <div class="flex items-center space-x-3">
                <i class="fas fa-file-csv text-green-600"></i>
                <div class="text-left">
                  <div class="font-medium text-gray-800">CSV Alerts</div>
                  <div class="text-xs text-gray-500">Alert timeline data</div>
                </div>
              </div>
              <i class="fas fa-download text-green-600"></i>
            </button>
            
            <button data-download="${sessionId}" data-format="pdf" 
                    class="download-btn flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors cursor-pointer">
              <div class="flex items-center space-x-3">
                <i class="fas fa-file-pdf text-red-600"></i>
                <div class="text-left">
                  <div class="font-medium text-gray-800">PDF Report</div>
                  <div class="text-xs text-gray-500">Formatted report</div>
                </div>
              </div>
              <i class="fas fa-download text-red-600"></i>
            </button>
          </div>
          
          <div class="mt-4 p-3 bg-gray-50 rounded-lg">
            <div class="text-xs text-gray-600">
              <i class="fas fa-info-circle mr-1"></i>
              Session ID: <code class="bg-white px-1 rounded">${sessionId}</code>
            </div>
          </div>
        </div>
      `
      
      // Add event listeners to download buttons
      setTimeout(() => {
        const downloadButtons = exportSection.querySelectorAll('.download-btn')
        downloadButtons.forEach(button => {
          button.addEventListener('click', (e) => {
            e.preventDefault()
            const sessionId = button.dataset.download
            const format = button.dataset.format
            console.log('Download button clicked:', sessionId, format)
            this.downloadReport(sessionId, format)
          })
        })
      }, 100)
    }
  },
  
  async downloadReport(sessionId, format) {
    console.log(`Download requested for session: ${sessionId}, format: ${format}`)
    
    try {
      if (utils && utils.showLoading) {
        utils.showLoading(true)
      }
      
      const downloadUrl = `http://localhost:8000/export/${sessionId}/${format}`
      console.log(`Downloading from: ${downloadUrl}`)
      
      const response = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Accept': '*/*'
        }
      })
      
      console.log(`Response status: ${response.status}`)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error(`Download failed with status ${response.status}: ${errorText}`)
        throw new Error(`Failed to download ${format.toUpperCase()} report: ${response.status} ${response.statusText}`)
      }
      
      const blob = await response.blob()
      console.log(`Blob created, size: ${blob.size} bytes`)
      
      // Determine file extension based on format and content type
      let fileExtension = format
      if (format === 'pdf') {
        fileExtension = 'html' // Our PDF is actually HTML
      }
      
      // Get filename from response headers or create one
      const contentDisposition = response.headers.get('content-disposition')
      let filename = `session_report_${sessionId}.${fileExtension}`
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;\s]+)/)
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/"/g, '')
        }
      }
      
      console.log(`Downloading as filename: ${filename}`)
      
      // Create download using URL.createObjectURL
      const url = window.URL.createObjectURL(blob)
      
      // Create temporary download link
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = filename
      
      // Add to DOM, click, and remove
      document.body.appendChild(a)
      a.click()
      
      // Clean up
      setTimeout(() => {
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }, 100)
      
      if (utils && utils.showLoading) {
        utils.showLoading(false)
      }
      
      if (utils && utils.showToast) {
        utils.showToast('Download Complete', `${format.toUpperCase()} report downloaded successfully`, 'success')
      }
      
      console.log('Download completed successfully')
      
    } catch (error) {
      console.error('Download error:', error)
      
      if (utils && utils.showLoading) {
        utils.showLoading(false)
      }
      
      if (utils && utils.showToast) {
        utils.showToast('Download Failed', error.message, 'error')
      } else {
        alert(`Download failed: ${error.message}`)
      }
    }
  }
}

// Settings Management
const settingsManager = {
  init() {
    // Update range value displays
    elements.focusTimeout?.addEventListener('input', (e) => {
      e.target.nextElementSibling.textContent = `${e.target.value}s`
    })
    
    elements.faceTimeout?.addEventListener('input', (e) => {
      e.target.nextElementSibling.textContent = `${e.target.value}s`
    })
    
    elements.confidenceThreshold?.addEventListener('input', (e) => {
      e.target.nextElementSibling.textContent = e.target.value
    })
  }
}

// Event Listeners
const initializeEventListeners = () => {
  // Session form submission
  elements.sessionForm?.addEventListener('submit', (e) => {
    e.preventDefault()
    const candidateName = elements.candidateName.value.trim()
    const sessionType = elements.sessionType.value
    
    if (candidateName) {
      sessionManager.startSession(candidateName, sessionType)
    }
  })
  
  // Session controls
  elements.pauseSession?.addEventListener('click', () => {
    sessionManager.pauseSession()
  })
  
  elements.stopSession?.addEventListener('click', () => {
    if (confirm('Are you sure you want to end this session?')) {
      sessionManager.stopSession()
    }
  })
  
  // Settings button (placeholder)
  document.getElementById('settingsBtn')?.addEventListener('click', () => {
    utils.showToast('Settings', 'Settings panel coming soon!', 'info')
  })
  
  // Theme toggle (placeholder)
  document.getElementById('themeToggle')?.addEventListener('click', () => {
    utils.showToast('Theme', 'Dark mode coming soon!', 'info')
  })
}

// Session History Management
const sessionHistory = {
  async loadCompletedSessions() {
    try {
      const response = await fetch('http://localhost:8000/sessions/completed')
      if (!response.ok) {
        throw new Error('Failed to load session history')
      }
      
      const data = await response.json()
      this.displaySessionHistory(data.sessions)
      
    } catch (error) {
      console.error('Error loading session history:', error)
      const historySection = document.getElementById('sessionHistory')
      if (historySection) {
        historySection.innerHTML = `
          <div class="flex flex-col items-center justify-center py-12 text-gray-500">
            <i class="fas fa-exclamation-triangle text-4xl mb-4 text-yellow-500"></i>
            <p class="text-lg font-medium">Failed to load session history</p>
            <button onclick="sessionHistory.loadCompletedSessions()" class="mt-2 text-blue-600 hover:text-blue-800">
              <i class="fas fa-redo mr-1"></i>Try again
            </button>
          </div>
        `
      }
    }
  },
  
  displaySessionHistory(sessions) {
    const historySection = document.getElementById('sessionHistory')
    if (!historySection) return
    
    if (!sessions || sessions.length === 0) {
      historySection.innerHTML = `
        <div class="flex flex-col items-center justify-center py-12 text-gray-500">
          <i class="fas fa-folder-open text-4xl mb-4"></i>
          <p class="text-lg font-medium">No previous sessions</p>
          <p class="text-sm">Complete an interview to see session history</p>
        </div>
      `
      return
    }
    
    const sessionsHtml = sessions.map(session => {
      const riskColors = {
        low: 'bg-green-100 text-green-800',
        medium: 'bg-yellow-100 text-yellow-800', 
        high: 'bg-red-100 text-red-800',
        critical: 'bg-red-200 text-red-900'
      }
      
      const riskColor = riskColors[session.risk_level] || 'bg-gray-100 text-gray-800'
      const startDate = new Date(session.start_time).toLocaleDateString()
      const startTime = new Date(session.start_time).toLocaleTimeString()
      
      return `
        <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
          <div class="flex justify-between items-start mb-3">
            <div>
              <h5 class="font-semibold text-gray-800">${session.candidate_name}</h5>
              <p class="text-xs text-gray-500">${session.session_type} Interview</p>
            </div>
            <span class="inline-block px-2 py-1 rounded-full text-xs font-medium ${riskColor}">
              ${session.risk_level.charAt(0).toUpperCase() + session.risk_level.slice(1)}
            </span>
          </div>
          
          <div class="space-y-1 text-xs text-gray-600 mb-3">
            <div><i class="fas fa-calendar mr-1"></i>${startDate} at ${startTime}</div>
            <div><i class="fas fa-clock mr-1"></i>${session.duration_minutes.toFixed(1)} minutes</div>
            <div><i class="fas fa-exclamation-triangle mr-1"></i>${session.total_violations} violations</div>
          </div>
          
          <div class="flex space-x-2">
            <button data-download="${session.session_id}" data-format="json" 
                    class="history-download-btn flex-1 text-xs bg-blue-50 text-blue-700 py-2 px-3 rounded hover:bg-blue-100 transition-colors cursor-pointer">
              <i class="fas fa-download mr-1"></i>JSON
            </button>
            <button data-download="${session.session_id}" data-format="csv" 
                    class="history-download-btn flex-1 text-xs bg-green-50 text-green-700 py-2 px-3 rounded hover:bg-green-100 transition-colors cursor-pointer">
              <i class="fas fa-download mr-1"></i>CSV
            </button>
            <button data-download="${session.session_id}" data-format="pdf" 
                    class="history-download-btn flex-1 text-xs bg-red-50 text-red-700 py-2 px-3 rounded hover:bg-red-100 transition-colors cursor-pointer">
              <i class="fas fa-download mr-1"></i>PDF
            </button>
          </div>
        </div>
      `
    }).join('')
    
    historySection.innerHTML = `
      <div class="space-y-3 max-h-80 overflow-y-auto">
        ${sessionsHtml}
      </div>
    `
    
    // Add event listeners to history download buttons
    setTimeout(() => {
      const historyDownloadButtons = historySection.querySelectorAll('.history-download-btn')
      historyDownloadButtons.forEach(button => {
        button.addEventListener('click', (e) => {
          e.preventDefault()
          const sessionId = button.dataset.download
          const format = button.dataset.format
          console.log('History download button clicked:', sessionId, format)
          this.downloadSessionReport(sessionId, format)
        })
      })
    }, 100)
  },
  
  async downloadSessionReport(sessionId, format) {
    await sessionManager.downloadReport(sessionId, format)
  }
}

// Application Initialization
const initializeApp = () => {
  console.log('🚀 AI Proctor - Initializing...')
  
  // Check for required APIs
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    utils.showToast('Browser Error', 'Your browser does not support camera access', 'error')
    return
  }
  
  // Initialize components
  initializeEventListeners()
  settingsManager.init()
  
  // Load session history
  sessionHistory.loadCompletedSessions()
  
  // Show welcome message
  utils.showToast('Welcome', 'AI Proctor system ready', 'success')
  
  console.log('✅ AI Proctor - Initialized successfully')
}

// Start the application
document.addEventListener('DOMContentLoaded', () => {
  initializeApp()
  
  // Make objects globally accessible for onclick handlers
  window.sessionManager = sessionManager
  window.sessionHistory = sessionHistory
  window.utils = utils
})

// Export for potential module usage
export { AppState, utils, videoProcessing, sessionManager, sessionHistory }
