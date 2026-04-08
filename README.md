# Negotiation Trainer - Strategic Business Deal Simulator

**Team:** Royal Ace  
**Hackathon:** Meta PyTorch OpenEnv AI Hackathon 2026

## Description
This environment simulates real-world business negotiations (salary talks, vendor contracts, client deals). 
The agent must make smart offers, balance multiple issues, maintain relationships, and handle bluffs.

## Tasks
- **basic_deal** (Easy): Simple price negotiation
- **multi_issue** (Medium): Price + timeline + extras
- **bluff_handling** (Hard): Detect and respond to bluffs

## Features
- Full OpenEnv compliance (typed models, step/reset/state)
- Dense reward with partial progress
- Deterministic graders (0.0 - 1.0)
- Docker support
- Baseline inference script

## How to Run Locally
```bash
pip install -r requirements.txt
python inference.py