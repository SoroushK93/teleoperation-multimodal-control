# Multimodal Robot Teleoperation Framework

This repository provides a unified framework for evaluating different teleoperation modalities:

- Vision-based hand tracking
- Keyboard control
- LLM-based natural language control

## Features
- Real-time robot control (UR10e)
- Multi-camera hand tracking (MediaPipe)
- Voice + LLM command parsing (GPT-4o)
- Safety-constrained motion execution
- Latency measurement tools

## Modalities

### 1. Hand Tracking
Camera-based control using hand motion and gestures.

### 2. Keyboard Control
Direct incremental Cartesian control using key bindings.

### 3. LLM Control
Voice commands → Whisper → GPT → structured robot actions.

## Setup

```bash
pip install -r requirements.txt

Notes
Requires URBasic for robot communication
Set your OpenAI API key:
export OPENAI_API_KEY=your_key_here
