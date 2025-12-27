# MusicGen Pro Studio – Unified Implementation Plan

> **Consolidation of**: `ENHANCEMENT_PLAN.md`, `PRO_STUDIO_PLAN.md`, `MUSIC_AI_TRAINING_REQUIREMENTS.md`, `INSTALL_NOTES.md`

---

## Executive Summary

Transform MusicGen Studio into a **professional AI DAW** with:

1. Multi-Agent LLM architecture for intelligent prompt enhancement.
2. Multi-Genre "Studio" pipelines (Chiptune, Hip Hop, Metal, Jazz, Classical).
3. RAG system for style-aware retrieval from your personal library.
4. Pro-level audio engineering (Upscaling via AudioSR, In-Painting via VampNet).
5. Robust SQLite-based library with Albums, Favorites, Comments.
6. Suno-inspired UI with waveform editing (Wavesurfer.js).

---

## Hardware Baseline (Validated)

| Component | Spec | Status |
|-----------|------|--------|
| GPU | RTX 3060 12GB | ✅ LoRA Training OK |
| RAM | 48GB DDR4 | ✅ Multi-LLM OK |
| Storage | 5.5TB SSD | ✅ Datasets OK |
| CPU | i5-14600K | ✅ Preprocessing OK |

---

## Phase 1: Foundation (✅ COMPLETE)

### 1.1 Core Pipeline

- [x] `scripts/preprocess.py`: Audio chunking to 30s @ 32kHz mono.
- [x] `scripts/autolabel.py`: BPM, Key, Mood detection via Essentia/Librosa.
- [x] `scripts/train.py`: MusicGen LoRA training with PEFT.
- [x] `scripts/generate.py`: Text-to-Music generation.

### 1.2 Web UI (Flask)

- [x] Dashboard, Upload, Process, Train, Generate, Library pages.
- [x] Suno-style dark theme with accent gradients.

---

## Phase 2: Multi-Agent Architecture (✅ COMPLETE)

### 2.1 LLM Engine

- [x] `llm/engine.py`: Llama.cpp integration for local LLM inference.
- [x] Uses Llama-3.2-3B (or Phi-3-mini) for CPU-based agents.

### 2.2 Agents

- [x] `agents/base_agent.py`: Base class with Llama-3 prompt formatting.
- [x] `agents/orchestrator.py`: Routes tasks to specialized agents.
- [x] **Prompt Agent**: Expands user prompts with musical terminology.

### 2.3 API Integration

- [x] `/api/agents/enhance-prompt`: AI-powered prompt refinement.

---

## Phase 3: Multi-Genre Studios (✅ COMPLETE)

### 3.1 Studio Manager

- [x] `studios/manager.py`: Lazy-loads specialized ensembles per genre.
- [x] Hot-swapping between studios to conserve RAM.

### 3.2 Implemented Studios

| Studio | Ensemble | Specialized Agents |
|--------|----------|-------------------|
| **Chiptune** | `ChiptuneEnsemble` | LakhNESWrapper (structure), ChiptuneRNN (melody) |
| **Cypher** (Hip Hop) | `CypherEnsemble` | BeatMakerWrapper (drums), FlowGenWrapper (lyrics) |
| **Ironclad** (Metal) | `IroncladEnsemble` | ShredGPWrapper (guitar), MetalDrummerWrapper (drums) |
| **Blue Note** (Jazz) | Placeholder | DeepJazz (harmony), JazzLoRA (style) |
| **Virtuoso** (Classical) | Placeholder | NotaGen (score), OrchestralLoRA (style) |

### 3.3 API Integration

- [x] `/api/studio/select`: Switch active studio.
- [x] `/api/studio/current`: Get current studio.
- [x] `/api/generate`: Routes through active studio's ensemble.

---

## Phase 4: RAG System (✅ COMPLETE)

### 4.1 Vector Database

- [x] `rag/system.py`: ChromaDB + SentenceTransformer embeddings.
- [x] Indexes `metadata/*.json` for semantic search.

### 4.2 Features

- [x] Find similar tracks by prompt.
- [x] Inject "Style Ref" into generation prompts.

### 4.3 API

- [x] `/api/rag/similar`: Query for similar tracks.
- [x] `/api/rag/index`: Trigger background re-indexing.

---

## Phase 5: Pro Audio Engineering (🔄 IN PROGRESS)

### 5.1 Engineering Layer

- [ ] `engineering/upscaler.py`: AudioSR for 32kHz → 48kHz+ upscaling.
- [ ] `engineering/editor.py`: VampNet for in-painting (with MusicGen fallback).

### 5.2 In-Painting Workflow ("Cut & Stitch")

1. User selects region [10s–15s] in waveform editor.
2. Backend slices audio into Part A (0–10s) and Part C (15–30s).
3. Generates Part B (new 5s) using Part A as context.
4. **Primary**: VampNet true in-painting (preserves ending).
5. **Fallback**: MusicGen continuation + crossfade.

### 5.3 Remix Controls (Sliders)

| Control | Parameter | Effect |
|---------|-----------|--------|
| Melody Loyalty | Chroma strength | 0% = Ignore melody, 100% = Exact |
| Weirdness | Temperature | 0.1 = Safe, 1.5 = Chaos |
| Genre Shift | CFG + Prompt weight | How much prompt overrides audio |
| Structure Consistency | Bar alignment | Enforce beat grids |

---

## Phase 6: Robust Library (🔄 IN PROGRESS)

### 6.1 Database

- [x] `database/schema.py`: SQLAlchemy models (Track, Album, Comment).
- [x] `database/manager.py`: CRUD operations.
- [ ] `database/migrate.py`: Import existing `metadata/*.json` to SQLite.

### 6.2 Features

| Feature | Description |
|---------|-------------|
| **Favorites** | ❤️ Toggle on tracks |
| **Disliked/Trash** | 👎 Hide from main view |
| **Albums** | Group tracks into collections |
| **Comments** | Time-stamped notes on tracks |
| **Batch Edit** | Select multiple → Add to Album / Delete |

### 6.3 API Endpoints (Planned)

- `/api/library/tracks` (GET, POST)
- `/api/library/tracks/<id>/favorite` (POST)
- `/api/library/albums` (GET, POST)
- `/api/library/tracks/<id>/comments` (GET, POST)

---

## Phase 7: Waveform Editor UI (⏳ TODO)

### 7.1 Technology

- **Wavesurfer.js** + Regions Plugin for selection.

### 7.2 Interactions

| Action | Trigger |
|--------|---------|
| Select Region | Drag on waveform |
| Audition Loop | Click "Loop" to preview selection |
| Regenerate | Right-click → "Regenerate" (triggers In-Painting) |
| Extend | Right-click → "Extend" (triggers continuation) |

---

## Phase 8: Advanced Features (⏳ TODO)

| Feature | Priority | Description |
|---------|----------|-------------|
| Stem Separation | High | Demucs integration for drums/bass/vocals |
| DAW Export | Medium | Ableton/FL project file export |
| MIDI Export | Medium | Extract melodic content as MIDI |
| Voice Input | Low | Whisper transcription for prompts |
| VST Plugin | Low | Load MusicGen as DAW instrument |

---

## Execution Order (Sprints)

| Sprint | Focus | Status |
|--------|-------|--------|
| 1 | Core Pipeline + Web UI | ✅ Done |
| 2 | Multi-Agent + Studios | ✅ Done |
| 3 | RAG System | ✅ Done |
| 4 | Library DB + Migration | 🔄 In Progress |
| 5 | Engineering (Upscaler, Editor) | ⏳ Next |
| 6 | Waveform UI + In-Painting | ⏳ Planned |
| 7 | Stem Separation + Export | ⏳ Backlog |

---

## Model Downloads Required

| Model | Size | Purpose |
|-------|------|---------|
| Llama-3.2-3B-GGUF | ~2GB | Agents (Prompt, Quality) |
| MusicGen-Medium | ~1.5GB | Core generation |
| CLAP | ~600MB | Audio embeddings (RAG) |
| AudioSR | ~1GB | Super-resolution upscaling |
| VampNet | ~500MB | Audio in-painting |
| Demucs | ~500MB | Stem separation |

---

## Files to Deprecate

| Old File | Reason |
|----------|--------|
| `ENHANCEMENT_PLAN.md` | Merged into this plan |
| `PRO_STUDIO_PLAN.md` | Merged into this plan |
| `MUSIC_AI_TRAINING_REQUIREMENTS.md` | Merged (HW section) |
| `INSTALL_NOTES.md` | Keep as README or merge |
