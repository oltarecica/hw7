# HW7 Machine Learning API - Olta Recica & Olsa Berani

This project creates a simple FastAPI service that serves a Machine Learning model trained on the Iris dataset.

## Files
- `model_training.ipynb`: trains the model and saves it as `model.pkl`
- `model.pkl`: serialized model file
- `api.py`: FastAPI application that loads the model and provides endpoints
- `example.json`: example input for prediction
- `client_request.py`: sends a single data point to the API
- `client_file_request.py`: sends a JSON file to the API

## How to run

1. Start the API:
   ```bash
   uv run uvicorn api:app --reload --port 8000
