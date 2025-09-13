from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import shutil
from datetime import datetime
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create static directory at startup
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML content for the webpage
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>EEG Prediction Pipeline</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="bg-white p-6 rounded-lg shadow-lg w-full max-w-2xl">
        <h1 class="text-2xl font-bold mb-4 text-center">EEG Prediction Pipeline</h1>
        <button id="startBtn" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full mb-4">
            Start Prediction
        </button>
        <h2 class="text-lg font-semibold mb-2">Terminal Output</h2>
        <pre id="output" class="bg-gray-900 text-white p-4 rounded-lg h-64 overflow-auto"></pre>
        <div id="results" class="mt-4"></div>
    </div>
    <script>
        document.getElementById('startBtn').addEventListener('click', async () => {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').textContent = 'Running...';
            document.getElementById('output').textContent = 'Starting prediction...';
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/run-pipeline', { method: 'POST' });
                const result = await response.json();
                document.getElementById('output').textContent = result.output;

                if (result.png_file) {
                    const imgLink = document.createElement('a');
                    imgLink.href = `/static/${result.png_file}`;
                    imgLink.textContent = 'View EEG Plot (PNG)';
                    imgLink.className = 'text-blue-500 hover:underline mr-4';
                    document.getElementById('results').appendChild(imgLink);
                }
                if (result.txt_file) {
                    const txtLink = document.createElement('a');
                    txtLink.href = `/static/${result.txt_file}`;
                    txtLink.textContent = 'View EEG Analysis Report (TXT)';
                    txtLink.className = 'text-blue-500 hover:underline';
                    document.getElementById('results').appendChild(txtLink);
                }
            } catch (error) {
                document.getElementById('output').textContent = 'Error: ' + error.message;
            } finally {
                document.getElementById('startBtn').disabled = false;
                document.getElementById('startBtn').textContent = 'Start Prediction';
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=html_content)

@app.post("/run-pipeline")
async def run_pipeline():
    try:
        logger.debug("Starting pipeline execution")
        # Run the real_pipeline.py script with UTF-8 encoding
        process = subprocess.run(
            ["python", "real_pipeline.py"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        output = process.stdout + process.stderr
        logger.debug(f"Pipeline output: {output}")

        # Find the latest PNG and text files in the data directory
        data_dir = "data"
        png_file = None
        txt_file = None
        if os.path.exists(data_dir):
            png_files = sorted(
                glob.glob(os.path.join(data_dir, "eeg_plot_*.png")),
                key=os.path.getmtime,
                reverse=True
            )
            txt_files = sorted(
                glob.glob(os.path.join(data_dir, "eeg_analysis_report_*.txt")),
                key=os.path.getmtime,
                reverse=True
            )

            if png_files:
                latest_png = os.path.basename(png_files[0])
                shutil.copy(os.path.join(data_dir, latest_png), os.path.join("static", latest_png))
                png_file = latest_png

            if txt_files:
                latest_txt = os.path.basename(txt_files[0])
                shutil.copy(os.path.join(data_dir, latest_txt), os.path.join("static", latest_txt))
                txt_file = latest_txt

        return {
            "output": output,
            "png_file": png_file,
            "txt_file": txt_file
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Script error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Script error: {e.stderr}")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")