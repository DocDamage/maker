/**
 * MusicGen Studio - Frontend JavaScript
 */

// ============================================
// Navigation
// ============================================

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const section = item.dataset.section;

        // Update nav
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');

        // Update sections
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(section).classList.add('active');

        // Refresh data for certain sections
        if (section === 'library') loadGeneratedLibrary();
        if (section === 'process') loadMetadata();
    });
});

// ============================================
// API Helpers
// ============================================

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`/api${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast('Connection error', 'error');
        throw error;
    }
}

// ============================================
// Status Updates
// ============================================

async function refreshStatus() {
    try {
        const status = await api('/status');

        // Update stats
        document.getElementById('rawCount').textContent = status.raw_audio_count;
        document.getElementById('processedCount').textContent = status.processed_count;
        document.getElementById('labeledCount').textContent = status.labeled_count;
        document.getElementById('generatedCount').textContent = status.generated_count;
        document.getElementById('modelName').textContent = status.model_name?.split('/').pop() || 'Not loaded';

        // Update pipeline status indicators
        updatePipelineStatus(status);

        // Update progress bars
        updateAllProgress(status.progress);

    } catch (error) {
        console.error('Status refresh failed:', error);
    }
}

function updatePipelineStatus(status) {
    const steps = ['upload', 'process', 'label', 'train', 'generate'];

    if (status.raw_audio_count > 0) {
        document.getElementById('uploadStatus').textContent = 'Complete';
        document.getElementById('step-upload').classList.add('complete');
    }

    if (status.processed_count > 0) {
        document.getElementById('processStatus').textContent = 'Complete';
        document.getElementById('step-process').classList.add('complete');
    }

    if (status.labeled_count > 0) {
        document.getElementById('labelStatus').textContent = 'Complete';
        document.getElementById('step-label').classList.add('complete');
    }

    if (status.latest_lora) {
        document.getElementById('trainStatus').textContent = 'Complete';
        document.getElementById('step-train').classList.add('complete');
    }
}

function updateAllProgress(progress) {
    // Preprocessing
    if (progress.preprocessing.status === 'running') {
        document.getElementById('preprocessProgressBar').style.width = `${progress.preprocessing.progress}%`;
        document.getElementById('preprocessMessage').textContent = progress.preprocessing.message;
        document.getElementById('btnPreprocess').disabled = true;
    } else if (progress.preprocessing.status === 'complete') {
        document.getElementById('preprocessProgressBar').style.width = '100%';
        document.getElementById('preprocessMessage').textContent = progress.preprocessing.message;
        document.getElementById('btnPreprocess').disabled = false;
        document.getElementById('btnPreprocess').textContent = 'Reprocess';
    } else if (progress.preprocessing.status === 'error') {
        document.getElementById('preprocessMessage').textContent = '❌ ' + progress.preprocessing.message;
        document.getElementById('btnPreprocess').disabled = false;
    }

    // Labeling
    if (progress.labeling.status === 'running') {
        document.getElementById('labelProgressBar').style.width = `${progress.labeling.progress}%`;
        document.getElementById('labelMessage').textContent = progress.labeling.message;
        document.getElementById('btnLabel').disabled = true;
    } else if (progress.labeling.status === 'complete') {
        document.getElementById('labelProgressBar').style.width = '100%';
        document.getElementById('labelMessage').textContent = progress.labeling.message;
        document.getElementById('btnLabel').disabled = false;
        document.getElementById('btnLabel').textContent = 'Re-label';
    } else if (progress.labeling.status === 'error') {
        document.getElementById('labelMessage').textContent = '❌ ' + progress.labeling.message;
        document.getElementById('btnLabel').disabled = false;
    }

    // Training
    if (progress.training.status === 'running') {
        document.getElementById('trainProgressBar').style.width = `${progress.training.progress}%`;
        document.getElementById('trainStatusValue').textContent = 'Training...';
        document.getElementById('trainEpoch').textContent = `${progress.training.epoch}`;
        document.getElementById('trainLoss').textContent = progress.training.loss.toFixed(4);
        document.getElementById('trainLog').textContent = progress.training.message;
        document.getElementById('btnTrain').disabled = true;
    } else if (progress.training.status === 'complete') {
        document.getElementById('trainProgressBar').style.width = '100%';
        document.getElementById('trainStatusValue').textContent = 'Complete';
        document.getElementById('btnTrain').disabled = false;
        showToast('Training complete!', 'success');
    } else if (progress.training.status === 'error') {
        document.getElementById('trainStatusValue').textContent = 'Error';
        document.getElementById('trainLog').textContent = progress.training.message;
        document.getElementById('btnTrain').disabled = false;
    }

    // Generation
    if (progress.generating.status === 'running') {
        document.getElementById('generateProgress').style.display = 'block';
        document.getElementById('generateProgressBar').style.width = `${progress.generating.progress}%`;
        document.getElementById('generateMessage').textContent = progress.generating.message;
        document.getElementById('btnGenerate').disabled = true;
    } else if (progress.generating.status === 'complete') {
        document.getElementById('generateProgress').style.display = 'none';
        document.getElementById('btnGenerate').disabled = false;
        showToast('Music generated!', 'success');
        loadGeneratedLibrary();
    } else if (progress.generating.status === 'error') {
        document.getElementById('generateProgress').style.display = 'none';
        document.getElementById('generateMessage').textContent = 'Generation failed';
        document.getElementById('btnGenerate').disabled = false;
        showToast('Generation failed: ' + progress.generating.message, 'error');
    }
}

// Poll for updates every 2 seconds
setInterval(refreshStatus, 2000);

// ============================================
// File Upload
// ============================================

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

async function handleFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('uploadProgressBar').style.width = '50%';
    document.getElementById('uploadProgressText').textContent = `Uploading ${files.length} file(s)...`;

    try {
        const result = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await result.json();

        document.getElementById('uploadProgressBar').style.width = '100%';
        document.getElementById('uploadProgressText').textContent = `Uploaded ${data.count} file(s)`;

        showToast(`Uploaded ${data.count} file(s)`, 'success');

        setTimeout(() => {
            document.getElementById('uploadProgress').style.display = 'none';
            loadRawFiles();
            refreshStatus();
        }, 1500);

    } catch (error) {
        document.getElementById('uploadProgressText').textContent = 'Upload failed';
        showToast('Upload failed', 'error');
    }
}

async function loadRawFiles() {
    const files = await api('/files/raw');
    const container = document.getElementById('rawFileList');

    container.innerHTML = files.map(f => `
        <div class="file-item">
            <span class="file-icon">🎵</span>
            <div class="file-info">
                <div class="file-name" title="${f.name}">${f.name}</div>
                <div class="file-size">${formatFileSize(f.size)}</div>
            </div>
        </div>
    `).join('');
}

// ============================================
// Pipeline Actions
// ============================================

async function startPreprocessing() {
    showToast('Starting preprocessing...', 'info');
    await api('/preprocess', { method: 'POST' });
}

async function startLabeling() {
    showToast('Starting auto-labeling...', 'info');
    await api('/label', { method: 'POST' });
}

async function startTraining() {
    const epochs = parseInt(document.getElementById('epochs').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);

    showToast('Starting training...', 'info');
    document.getElementById('trainLog').textContent = 'Initializing...\n';

    await api('/train', {
        method: 'POST',
        body: JSON.stringify({ epochs, learning_rate: learningRate }),
    });
}

async function startGeneration() {
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) {
        showToast('Please enter a description', 'error');
        return;
    }

    const duration = parseInt(document.getElementById('duration').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const count = parseInt(document.getElementById('count').value);

    showToast('Starting generation...', 'info');

    await api('/generate', {
        method: 'POST',
        body: JSON.stringify({ prompt, duration, temperature, count }),
    });
}

function setPrompt(text) {
    document.getElementById('prompt').value = text;
}

// Temperature slider display
document.getElementById('temperature').addEventListener('input', (e) => {
    document.getElementById('temperatureValue').textContent = e.target.value;
});

// ============================================
// Metadata & Library
// ============================================

async function loadMetadata() {
    const metadata = await api('/files/metadata');
    const container = document.getElementById('metadataList');

    container.innerHTML = metadata.slice(0, 12).map(m => `
        <div class="metadata-item">
            <h4>${m.filename || 'Unknown'}</h4>
            <p>"${m.description}"</p>
            <div class="metadata-tags">
                ${m.bpm ? `<span class="metadata-tag">🎵 ${Math.round(m.bpm)} BPM</span>` : ''}
                ${m.key ? `<span class="metadata-tag">🎹 ${m.key} ${m.mode || ''}</span>` : ''}
                ${m.mood ? `<span class="metadata-tag">💫 ${m.mood}</span>` : ''}
                ${m.genre ? `<span class="metadata-tag">🎸 ${m.genre}</span>` : ''}
            </div>
        </div>
    `).join('');
}

async function loadGeneratedLibrary() {
    const files = await api('/files/generated');
    const container = document.getElementById('generatedLibrary');

    if (files.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>No generated tracks yet. Create some music!</p>
            </div>
        `;
        return;
    }

    container.innerHTML = files.map(f => `
        <div class="audio-card">
            <div class="audio-card-header">
                <div class="audio-card-icon">🎵</div>
                <div class="audio-card-info">
                    <h3>${f.name}</h3>
                    <span>${formatFileSize(f.size)} • ${formatDate(f.modified)}</span>
                </div>
            </div>
            <audio controls src="${f.url}" preload="none"></audio>
        </div>
    `).join('');
}

// ============================================
// Utilities
// ============================================

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDate(isoString) {
    return new Date(isoString).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    refreshStatus();
    loadRawFiles();
});
