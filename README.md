# Breast Cancer Detection AI Platform 🏥

An advanced AI-powered medical imaging SaaS platform for breast ultrasound analysis. This project combines **Attention U-Net** for precise lesion segmentation and **CNN** for classification (Normal, Benign, Malignant), providing a professional diagnostic tool with automated report generation.

![Medical Imaging](https://img.shields.io/badge/Medical-AI-blue)
![React](https://img.shields.io/badge/Frontend-React-61DAFB)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-05998B)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

- **🎯 Precision Segmentation**: Uses an Attention-based U-Net architecture to isolate lesions in ultrasound images.
- **🧠 Accurate Classification**: Classifies tissue into three categories: *Normal*, *Benign*, and *Malignant*.
- **📊 Real-time Analysis**: Provides instant confidence scores, area coverage metrics, and visual overlays.
- **📋 Clinical Explanations**: AI-generated reports explaining the findings with clinical terminology.
- **📁 History Management**: Stores analysis results in a MongoDB database for future reference.
- **🖨️ PDF Report Generation**: One-click download of professional, hospital-style medical reports including images and metrics.
- **✨ Premium UI**: Responsive dark-mode interface with glassmorphism and modern medical aesthetics.

---

## 🛠️ Technology Stack

### Backend
- **Core**: Python 3.11+, FastAPI
- **Deep Learning**: PyTorch, Torchvision
- **Image Processing**: OpenCV, Pillow, Albumentations
- **PDF Generation**: ReportLab
- **Database**: MongoDB (Motor / Async driver)
- **Logging**: Python-Loguru style custom logging

### Frontend
- **Framework**: React.js
- **Styling**: Vanilla CSS (Custom Variable System)
- **State Management**: React Hooks
- **API Communication**: Axios

---

## 🏗️ Project Structure

```text
e:/final_year/
├── project/
│   ├── env/                # Python Virtual Environment
│   └── project/            # Main Source Code
│       ├── backend/        # FastAPI Server, Models, Services
│       ├── frontend/       # React App
│       ├── weights/        # AI Model Weights (.pth files)
│       ├── uploads/        # Input Images
│       └── outputs/        # Generated Masks and Overlays
└── .gitignore              # Root Git exclusion rules
```

---

## 🏁 Getting Started

### Prerequisites
- Python 3.11+
- Node.js & npm
- MongoDB (running locally or via Docker)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Namanverma991/breast-cancer-ai.git
   cd breast-cancer-ai
   ```

2. **Setup Backend**
   ```bash
   cd project/project
   python -m venv ../env
   source ../env/bin/activate  # On Windows: ..\env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   ```

### Running the Platform

1. **Start Backend** (from `project/project`)
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

2. **Start Frontend** (from `project/project/frontend`)
   ```bash
   npm start
   ```

The app will be available at `http://localhost:3000`.

---

## 📝 Medical Disclaimer

This application is intended for **research and educational purposes only**. It is an AI-assisted tool and **not** a substitute for professional clinical judgment, diagnosis, or treatment. Always consult with a qualified healthcare professional.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Developed with ❤️ for the Final Year Project.
