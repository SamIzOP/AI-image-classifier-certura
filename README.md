# AI Image Classifier

A web app that identifies objects in images using AI. Upload any photo and get instant predictions with confidence scores.

## Setup

1. **Install dependencies:**
```bash
pip install flask tensorflow pillow numpy
```

2. **Create project structure:**
```
project/
├── web_app.py
└── templates/
    └── interface.html
```

3. **Run the app:**
```bash
python web_app.py
```

4. **Open in browser:** `http://localhost:5000`

## How to Use

1. Upload an image (drag & drop or click to browse)
2. Preview your image
3. Click "Classify Image" 
4. View top 5 AI predictions with confidence percentages

## Features

- Supports JPG, PNG, GIF files (max 16MB)
- Drag & drop interface
- Real-time image preview
- Mobile-friendly design

## Requirements

- Python 3.7+
- Internet connection (downloads AI model on first run)

## Troubleshooting

- Make sure `interface.html` is in the `templates/` folder
- First run takes longer as it downloads the AI model (~14MB)
