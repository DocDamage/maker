# Pro Studio Implementation Plan (Sprint 4)

## Objective

Transform the "MusicGen Studio" into a professional-grade AI DAW ("Pro Studio") with robust library management, advanced audio engineering capabilities (upscaling, in-painting), and fine-grained control over generation parameters.

## 1. Technical Architecture: The "Engineering" Layer

We will introduce a new `engineering/` module to handle post-processing and advanced manipulation.

### A. The Engineering Agent (`engineering/agent.py`)

A specialized agent responsible for modifying existing audio.

- **Capabilities**:
  - **Upscaling**: Convert 32kHz generation to 48kHz+ using `AudioSR`.
  - **In-Painting**: Regenerate selected segments using `VampNet` (or MusicGen masking fallback).
  - **Remixing**: Generate variations using Melody Conditioning.
- **Sliders/Controls**:
  - **Melody Loyalty**: Controls how strictly the AI follows the original melody (via `chroma` extraction strength).
  - **Weirdness (Temperature)**: Controls the `temperature` parameter (0.1 = Safe, 1.5 = Chaos).
  - **Genre Shift**: Controls the balance between the Audio input and the Text Prompt (via `cfg_coef` and input strength).
  - **Structure Consistency**: Enforces bar/beat alignment (via rhythm constraints).

### B. Tooling Integration

1. **AudioSR (Super Resolution)**
    - Implement `engineering/upscaler.py`.
    - Asynchronous processing (background task) to prevent UI freezing.
2. **VampNet (In-Painting)**
    - Implement `engineering/editor.py`.
    - Function: `stitch_audio(before_audio, after_audio, prompt)`.
    - Uses cross-fading to blend the generated "middle" with the existing edges.

## 2. Robust Library Management (SQLite)

Transition from loose files to a structured Relational Database.

### A. Database Schema (`database/schema.py`) - *Drafted*

- **Tracks**: Stores filename, filepath, duration, seeds, prompts, configuration snapshot.
- **Albums**: Logical grouping of tracks.
- **Comments**: Time-stamped notes on tracks (e.g., "Fix the drums at 0:15").
- **Tags/Frieds**: `is_favorite`, `is_disliked` (trash), `is_public`.

### B. Migration Strategy

- `database/migrate.py`: Script to scan `metadata/*.json` and populate the SQLite DB on first run.

## 3. Frontend: Edit & Remaster Interface

A sophisticated waveform editor for the browser.

### A. Visual Editor

- **Technology**: `wavesurfer.js` + `regions-plugin`.
- **Interactions**:
  - **Select Region**: Drag to highlight a timeframe (e.g., 0:10 to 0:15).
  - **Context Menu**: Right-click region -> "Regenerate", "Delete", "Loop".
  - **Audition**: "Play Loop" button to preview the selection.

### B. Control Panel

- **Remix Section**:
  - Sliders for the new parameters (Melody Loyalty, Weirdness, Genre Shift).
  - "Batch Process" toggle.

## 4. Workflow: In-Painting (The "Cut & Stitch")

1. **User** selects Region [10s - 15s] in a 30s track.
2. **Frontend** sends: `track_id`, `start=10`, `end=15`, `prompt="Drum fill"`.
3. **Backend**:
    - Cuts audio into `Part A` (0-10s) and `Part C` (15-30s).
    - Uses `Part A` (last 5s) as the "Prompt Audio" for context.
    - Generates `Part B` (New 5s).
    - **Stitching**: Crossfades `A+B` and `B+C` (100ms overlap).
    - Returns new `Track` object.

## 5. Library Features

- **My Library Page**:
  - **Views**: Grid View (Cards) vs. List View (Metadata heavy).
  - **Filters**: Favorites, Albums, Date.
  - **Actions**:
    - **Heart/Dislike**: Instant toggle.
    - **Comment**: Add text notes.
    - **Batch**: Select multiple -> "Add to Album", "Delete".

## 6. Execution Order

1. **Database Refinement**: Finish `schema.py` and `manager.py`, write `migrate.py`.
2. **Engineering Backend**: Implement `upscaler.py` and `editor.py`.
3. **API Endpoints**: Connect Flask routes to these new engineering functions.
4. **Frontend**: Build the `wavesurfer` editor and connect the UI sliders.
