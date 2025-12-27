"""
MusicGen Studio - Web UI Backend

A Flask-based web server that provides a modern UI for the MusicGen training pipeline.

Usage:
    python app.py
    
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

from config import (
    RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR, METADATA_DIR, 
    LORA_DIR, OUTPUT_DIR, MODEL_NAME, AUDIO_EXTENSIONS
)

app = Flask(__name__, static_folder='ui/static', template_folder='ui/templates')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Global state for tracking progress
progress_state = {
    "preprocessing": {"status": "idle", "progress": 0, "message": "", "total": 0, "current": 0},
    "labeling": {"status": "idle", "progress": 0, "message": "", "total": 0, "current": 0},
    "training": {"status": "idle", "progress": 0, "message": "", "epoch": 0, "loss": 0},
    "generating": {"status": "idle", "progress": 0, "message": ""},
}

# Message queue for real-time updates
# Message queue for real-time updates
message_queues = {}

# =============================================================================
# AGENT INITIALIZATION
# =============================================================================
try:
    from agents.orchestrator import AgentOrchestrator
    
    # Path to where your models are stored
    MODELS_DIR = Path(r"C:\Users\dferr\Desktop\Aetherium_Game_Engine\local-ai\models\llm")
    
    orchestrator = AgentOrchestrator(MODELS_DIR)
    print("✅ Agent Orchestrator Initialized")
except Exception as e:
    orchestrator = None
    print(f"⚠️ Agent Orchestrator Failed to Load: {e}")

# Initialize Studio Manager
try:
    from studios.manager import StudioManager
    studio_manager = StudioManager()
    print("✅ Studio Manager Initialized")
except Exception as e:
    studio_manager = None
    print(f"⚠️ Studio Manager Failed to Load: {e}")

# Initialize RAG System
try:
    from rag.system import RAGSystem
    rag_system = RAGSystem(Path("data"))
    print("✅ RAG System Initialized")
except Exception as e:
    rag_system = None
    print(f"⚠️ RAG System Failed to Load: {e}")

# =============================================================================
# ROUTES - Pages
# =============================================================================


# =============================================================================
# ROUTES - Pages
# =============================================================================



# =============================================================================
# ROUTES - Pages
# =============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


# =============================================================================
# ROUTES - API Endpoints
# =============================================================================

@app.route('/api/status')
def get_status():
    """Get current system status and file counts."""
    raw_files = list(RAW_AUDIO_DIR.glob('*'))
    raw_audio = [f for f in raw_files if f.suffix.lower() in AUDIO_EXTENSIONS]
    
    processed_files = list(PROCESSED_AUDIO_DIR.glob('*.wav'))
    metadata_files = list(METADATA_DIR.glob('*.json'))
    
    # Find latest model
    lora_runs = sorted([d for d in LORA_DIR.iterdir() if d.is_dir()]) if LORA_DIR.exists() else []
    latest_model = str(lora_runs[-1]) if lora_runs else None
    
    output_files = list(OUTPUT_DIR.glob('*.wav')) if OUTPUT_DIR.exists() else []
    
    return jsonify({
        "raw_audio_count": len(raw_audio),
        "processed_count": len(processed_files),
        "labeled_count": len(metadata_files),
        "model_name": MODEL_NAME,
        "latest_lora": latest_model,
        "generated_count": len(output_files),
        "progress": progress_state,
    })


@app.route('/api/files/raw')
def get_raw_files():
    """List raw audio files."""
    files = []
    for f in RAW_AUDIO_DIR.glob('*'):
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return jsonify(sorted(files, key=lambda x: x['name']))


@app.route('/api/files/processed')
def get_processed_files():
    """List processed audio chunks."""
    files = []
    for f in PROCESSED_AUDIO_DIR.glob('*.wav'):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
        })
    return jsonify(sorted(files, key=lambda x: x['name']))


@app.route('/api/files/generated')
def get_generated_files():
    """List generated audio files."""
    files = []
    for f in OUTPUT_DIR.glob('*.wav'):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "url": f"/api/audio/generated/{f.name}",
        })
    return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True))


@app.route('/api/files/metadata')
def get_metadata():
    """Get all metadata/labels."""
    metadata = []
    for f in METADATA_DIR.glob('*.json'):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                metadata.append(data)
        except:
            pass
    return jsonify(metadata)


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload audio files."""
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    uploaded = []
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = RAW_AUDIO_DIR / filename
            file.save(str(filepath))
            uploaded.append(filename)
    
    return jsonify({"uploaded": uploaded, "count": len(uploaded)})


@app.route('/api/audio/generated/<filename>')
def serve_generated_audio(filename):
    """Serve a generated audio file."""
    return send_from_directory(OUTPUT_DIR, filename)


# =============================================================================
# =============================================================================
# ROUTES - Agent Actions
# =============================================================================

@app.route('/api/agents/enhance-prompt', methods=['POST'])
def enhance_prompt():
    """Enhance user prompt using local LLM agent."""
    if not orchestrator:
        return jsonify({"error": "Agent system not initialized (check logs)"}), 503
        
    data = request.json or {}
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    try:
        # Run agent
        result = orchestrator.enhance_prompt(prompt)
        return jsonify(result)
    except Exception as e:
        result = orchestrator.enhance_prompt(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/studio/select', methods=['POST'])
def select_studio():
    """Switch active studio."""
    if not studio_manager:
        return jsonify({"error": "Studio manager not initialized"}), 503
        
    data = request.json or {}
    studio = data.get('studio')
    
    if not studio:
        return jsonify({"error": "No studio specified"}), 400
        
    result = studio_manager.switch_studio(studio)
    return jsonify(result)

@app.route('/api/studio/current', methods=['GET'])
def get_current_studio():
    """Get active studio."""
    if not studio_manager:
        return jsonify({"studio": "standard", "error": "Not initialized"})
        
    return jsonify({"studio": studio_manager.get_active_studio()})


@app.route('/api/rag/similar', methods=['POST'])
def find_similar():
    """Find similar tracks via RAG."""
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 503
        
    data = request.json or {}
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    results = rag_system.query_similar(prompt)
    return jsonify(results)

@app.route('/api/rag/index', methods=['POST'])
def run_indexing():
    """Trigger re-indexing of library."""
    if not rag_system:
         return jsonify({"error": "RAG system not initialized"}), 503
         
    def background_index():
        rag_system.index_library(METADATA_DIR)
        
    thread = threading.Thread(target=background_index)
    thread.start()
    
    return jsonify({"status": "started", "message": "Indexing started in background"})


# =============================================================================
# ROUTES - Pipeline Actions
# =============================================================================

@app.route('/api/preprocess', methods=['POST'])
def start_preprocessing():
    """Start audio preprocessing."""
    def run_preprocessing():
        global progress_state
        progress_state["preprocessing"]["status"] = "running"
        progress_state["preprocessing"]["message"] = "Starting preprocessing..."
        
        try:
            # Import preprocessing module
            from scripts.preprocess import find_audio_files, process_file
            
            audio_files = find_audio_files(RAW_AUDIO_DIR)
            total = len(audio_files)
            progress_state["preprocessing"]["total"] = total
            
            for i, file_path in enumerate(audio_files):
                progress_state["preprocessing"]["current"] = i + 1
                progress_state["preprocessing"]["progress"] = int((i + 1) / total * 100)
                progress_state["preprocessing"]["message"] = f"Processing: {file_path.name}"
                
                process_file(file_path, PROCESSED_AUDIO_DIR)
            
            progress_state["preprocessing"]["status"] = "complete"
            progress_state["preprocessing"]["message"] = f"Processed {total} files"
            
        except Exception as e:
            progress_state["preprocessing"]["status"] = "error"
            progress_state["preprocessing"]["message"] = str(e)
    
    thread = threading.Thread(target=run_preprocessing)
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/label', methods=['POST'])
def start_labeling():
    """Start auto-labeling."""
    def run_labeling():
        global progress_state
        progress_state["labeling"]["status"] = "running"
        progress_state["labeling"]["message"] = "Starting auto-labeling..."
        
        try:
            from scripts.autolabel import analyze_file
            
            audio_files = list(PROCESSED_AUDIO_DIR.glob('*.wav'))
            total = len(audio_files)
            progress_state["labeling"]["total"] = total
            
            all_metadata = []
            
            for i, file_path in enumerate(audio_files):
                progress_state["labeling"]["current"] = i + 1
                progress_state["labeling"]["progress"] = int((i + 1) / total * 100)
                progress_state["labeling"]["message"] = f"Analyzing: {file_path.name}"
                
                analysis = analyze_file(file_path)
                all_metadata.append(analysis)
                
                # Save individual JSON
                json_path = METADATA_DIR / f"{file_path.stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, indent=2)
            
            # Save combined JSONL
            jsonl_path = METADATA_DIR / "data.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for analysis in all_metadata:
                    entry = {"path": analysis["path"], "description": analysis["description"]}
                    f.write(json.dumps(entry) + "\n")
            
            progress_state["labeling"]["status"] = "complete"
            progress_state["labeling"]["message"] = f"Labeled {total} files"
            
        except Exception as e:
            progress_state["labeling"]["status"] = "error"
            progress_state["labeling"]["message"] = str(e)
    
    thread = threading.Thread(target=run_labeling)
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training."""
    data = request.json or {}
    epochs = data.get('epochs', 10)
    learning_rate = data.get('learning_rate', 1e-4)
    
    def run_training():
        global progress_state
        progress_state["training"]["status"] = "running"
        progress_state["training"]["message"] = "Initializing training..."
        
        try:
            import subprocess
            
            cmd = [
                sys.executable, "scripts/train.py",
                "--epochs", str(epochs),
                "--lr", str(learning_rate),
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            for line in process.stdout:
                progress_state["training"]["message"] = line.strip()
                
                # Parse progress from output
                if "Epoch" in line:
                    try:
                        parts = line.split()
                        epoch_info = [p for p in parts if "/" in p][0]
                        current, total = epoch_info.split("/")
                        progress_state["training"]["epoch"] = int(current)
                        progress_state["training"]["progress"] = int(int(current) / int(total) * 100)
                    except:
                        pass
                if "loss" in line.lower():
                    try:
                        loss_val = float(line.split("loss")[-1].split()[0].strip(":"))
                        progress_state["training"]["loss"] = loss_val
                    except:
                        pass
            
            process.wait()
            
            if process.returncode == 0:
                progress_state["training"]["status"] = "complete"
                progress_state["training"]["message"] = "Training complete!"
            else:
                progress_state["training"]["status"] = "error"
                progress_state["training"]["message"] = "Training failed"
                
        except Exception as e:
            progress_state["training"]["status"] = "error"
            progress_state["training"]["message"] = str(e)
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/generate', methods=['POST'])
def start_generation():
    """Generate new music."""
    data = request.json or {}
    prompt = data.get('prompt', 'instrumental music')
    duration = data.get('duration', 30)
    count = data.get('count', 1)
    temperature = data.get('temperature', 1.0)
    
    def run_generation():
        global progress_state
        progress_state["generating"]["status"] = "running"
        progress_state["generating"]["message"] = "Loading model..."
        progress_state["generating"]["progress"] = 10
        
        try:
            # Route through Studio Manager if available
            final_prompt = prompt
            studio_metadata = {}
            
            if studio_manager:
                # This returns a dict with enhanced prompt and metadata
                studio_result = studio_manager.process_generation_request(prompt, duration)
                final_prompt = studio_result.get("prompt", prompt)
                studio_metadata = studio_result.get("metadata", {})
                
                # Update progress with studio info
                if "studio" in studio_metadata:
                    progress_state["generating"]["message"] = f"Using {studio_metadata['studio']} Ensemble..."
                    time.sleep(1) # Visual feedback
            
            # Use RAG to find similar tracks for style conditioning
            if rag_system:
                progress_state["generating"]["message"] = "Searching vector database..."
                similar_tracks = rag_system.query_similar(prompt, n_results=1)
                
                if similar_tracks:
                    ref_track = similar_tracks[0]['metadata']
                    ref_desc = ref_track.get('description', '')
                    # Append style reference to prompt
                    final_prompt += f" (Style Ref: {ref_desc})"
                    progress_state["generating"]["message"] = f"Found Ref: {ref_track.get('filename', 'Unknown')}"
                    time.sleep(0.5)

            import subprocess
            
            # We pass the enhanced prompt to the generation script
            cmd = [
                sys.executable, "scripts/generate.py",
                "--prompt", final_prompt,
                "--duration", str(duration),
                "--count", str(count),
                "--temperature", str(temperature),
            ]
            
            progress_state["generating"]["message"] = "Generating music..."
            progress_state["generating"]["progress"] = 30
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            for line in process.stdout:
                if "Saved" in line or "💾" in line:
                    progress_state["generating"]["progress"] = 90
                    progress_state["generating"]["message"] = line.strip()
            
            process.wait()
            
            if process.returncode == 0:
                progress_state["generating"]["status"] = "complete"
                progress_state["generating"]["progress"] = 100
                progress_state["generating"]["message"] = "Generation complete!"
            else:
                progress_state["generating"]["status"] = "error"
                progress_state["generating"]["message"] = "Generation failed"
                
        except Exception as e:
            progress_state["generating"]["status"] = "error"
            progress_state["generating"]["message"] = str(e)
    
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/progress/<task>')
def get_progress(task):
    """Get progress for a specific task."""
    if task in progress_state:
        return jsonify(progress_state[task])
    return jsonify({"error": "Unknown task"}), 404


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Create UI directories
    ui_dir = Path(__file__).parent / 'ui'
    (ui_dir / 'templates').mkdir(parents=True, exist_ok=True)
    (ui_dir / 'static' / 'css').mkdir(parents=True, exist_ok=True)
    (ui_dir / 'static' / 'js').mkdir(parents=True, exist_ok=True)
    
    print("\n🎵 MusicGen Studio")
    print("=" * 50)
    print(f"   Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
