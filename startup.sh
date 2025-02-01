#!/bin/bash

# Start FastAPI app
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit app
streamlit run streamlit_app.py --server.port 8501 &

# Keep the script running to ensure both services stay up
wait
