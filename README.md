# Poultry Bird Detection, Counting & Weight Estimation System

This repository contains a prototype AI system that processes fixed-camera poultry CCTV videos to:

- Detect and track birds  
- Maintain stable tracking IDs  
- Generate bird count over time  
- Estimate bird weight using visual proxy features  
- Produce annotated video output and structured JSON results  

---

## Project Overview

The system uses a **YOLOv8-based object detector** and **DeepSORT tracking** to identify individual birds and maintain their identities across frames. It then estimates bird weight using bounding-box area-based visual proxies and generates time-based bird count statistics.

---

## Input & Output Files

| File Type | Link |
|------------|------|
| Input Video | [https://drive.google.com/file/d/1pTVkCQVdMQhEtzxy9a9sQnRr05pYo_Tp/view](https://drive.google.com/file/d/1fS_mWrVnTCAbHfA17pQa2A_EPw6oL3au/view?usp=sharing) |
| Annotated Output Video | [https://drive.google.com/file/d/1g3xCjGkFtpAn3q6W_S_fZf2Im0lVZs0P/view](https://drive.google.com/file/d/1x2nCuVHT8qOHxtE_BSWsR8cvOIFaiF7r/view?usp=sharing) |
| Output JSON Response | [https://drive.google.com/file/d/1xSeLPqP9j9DBlAGi5cWnX8o2pK4DJknE/view](https://drive.google.com/file/d/1qbPQyo9EZNMm_AoL3XU4167_fwRGWtg5/view?usp=sharing) |

---

## Output Format

The system generates a JSON file with this structure:

{
"timestamp": "2025-12-24 15:15:08",
"bird_count": 214,
"avg_weight_index": 0.47,
"per_bird": [
{
"id": 12,
"bbox_area": 4220,
"weight_index": 0.45
}
]
}

---

## Installation

Create environment:

  python -m venv bird
  bird\Scripts\activate
  pip install -r requirements.txt


---

## Run Pipeline

  python app.py
  This will generate:
    outputs/annotated.mp4
    outputs/sample_response.json


---

## System Pipeline

1. YOLOv8 detects birds in each frame  
2. DeepSORT assigns persistent IDs  
3. Bounding box area used as weight proxy  
4. Frame-wise bird counts logged  
5. Annotated video & JSON exported  

---

## Weight Estimation Logic

Since real weight cannot be directly inferred visually, we use the following index:

\[
\text{weight\_index} = \frac{\text{bbox\_area}}{\text{max\_bbox\_area\_in\_frame}}
\]

This provides a **normalized relative weight proxy**, useful for trend analysis.

---

## Debugging Tools
python debug_detector.py


Shows live detection bounding boxes for tuning detection thresholds.


---

## 🚀 Technologies Used

- YOLOv8 (Ultralytics)  
- OpenCV  
- PyTorch  
- DeepSORT  
- NumPy  
- Python 3.11  

