# ALPR API - Automatic License Plate Recognition

This is a lightweight REST API for automatic license plate recognition, specifically designed to run on CPU-only environments like Render's free tier. It uses YOLOv8 for vehicle/plate detection and Tesseract for OCR.

## Related Projects
- **Frontend App**: [alpr-scanner-web](https://github.com/prpires66/alpr-scanner-web)

## Tech Stack
- **FastAPI**: Modern, fast (high-performance) web framework for building APIs with Python.
- **YOLOv8 (Ultralytics)**: Used for object detection (detecting the license plate region).
- **Tesseract OCR**: For character recognition on the detected plate.
- **OpenCV**: Image processing and manipulation.

## Core Functionality
- **`POST /read-plate`**: Receives an image file via `multipart/form-data`, detects the plate, performs OCR, and returns the plate number (formatted for Brazilian Mercosul standard `AAA1A23`) and raw text.
- **`GET /`**: Health check endpoint.

## Local Setup

### Prerequisites
- Python 3.10+
- **Tesseract OCR** installed on your system:
  - **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-por`

### Installation
1. Navigate to the `backend/` directory.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Deployment (Render)
This project is configured to be deployed on Render using the provided `Dockerfile`.
1. Create a new **Web Service** on Render.
2. Connect your repository.
3. Set the **Root Directory** to `backend`.
4. Render will automatically detect the Dockerfile and build the service.
5. Ensure your environment variables are set if necessary (default port is 8000).

## Model Information
The API uses a specialized YOLOv8 nano model (`license_plate_detector.pt`) which is automatically downloaded from Hugging Face during the Docker build process to keep the repository lightweight.
