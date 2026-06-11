# SSL for Side-Channel Attacks

This repository is for a remote research internship project on adapting self-supervised learning methods to side-channel attacks.

## Goal
Adapt the following self-supervised learning methods to side-channel analysis:
- TS2Vec
- SimCLR
- CPC (Contrastive Predicitive Coding) - https://github.com/Spijkervet/contrastive-predictive-coding
- MAE
- BYOL

## Initial Tasks
- Read the reference papers and repositories
- Explore ASCAD dataset format and baseline code
- Set up the development environment
- Run ASCAD code with ASCAD data
- Run TripletPower code and try adapting it to ASCAD data
- Prepare method-specific implementations and configs

## Repository Structure
- `data/`: datasets and processed traces
- `models/`: checkpoints, pretrained weights, outputs
- `configs/`: dataset, method, and training configs
- `notebooks/`: exploratory analysis and experiments
- `src/`: source code
- `weekly_updates/`: weekly markdown progress reports

## Weekly Updates
Progress will be tracked in markdown files under `weekly_updates/`.
