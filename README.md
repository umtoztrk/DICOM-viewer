# Medical Imaging Management System with Vector Database

## Project Overview

A comprehensive medical imaging management system designed for Eskişehir Osmangazi University's Computer Engineering Department. This system combines DICOM image processing with advanced vector database capabilities for efficient storage and retrieval of medical imaging data.

**Key Features**:
- DICOM image viewer with Hounsfield Unit conversion
- 28 specialized medical image filters
- Dual database architecture (Qdrant vector DB + SQLite)
- Patient/study/series hierarchical organization
- Radiology report generation
- Advanced search capabilities

## System Architecture
project/
├── app.py # Main application

├── Filters.py # 28 medical image processing algorithms

├── Qdrant.py # Vector database operations

├── SQLite.py # Relational database operations

├── UI/ # User interface components

│ ├── mainMenuUI.py # Main window

│ ├── editReportUI.py # Report editor

│ └── widgetFinalv2.py # DICOM tag viewer

└── requirements.txt # Dependencies


## Key Requirements Implemented

### Database Architecture (from Section 3.1.2)
- **Vector Database Collection**: 
  - Stores MR images, radiology reports, and patient data
  - Uses ResNet-50 for image embeddings (384-dimension vectors)
  - Maintains unique point IDs with non-null constraints
  - JSON-formatted DICOM metadata in payload

### Core Functionalities (from Section 3.1.1)
- DICOM image loading and display
- Patient data association and management
- Time-based and patient-name search
- Image acceptance/rejection workflow
- Report generation and export

### Quality Attributes (from Section 3.2.1)
- **Usability**: Intuitive PyQt5 interface
- **Security**: Patient data protection
- **Maintainability**: Modular design
- **Performance**: Optimized vector searches
- **Compatibility**: Works with hospital information systems

## Technologies Used

- **Python**: Primary programming language (v3.8+ recommended)
- **SQLite**: Relational database for patient metadata storage
- **Qdrant Vector Database**: High-performance vector similarity search engine for DICOM images
- **PyDICOM**: Python package for working with DICOM medical imaging files
- **PyQt5**: Cross-platform GUI toolkit for the user interface
- **OpenCV**: Medical image processing and transformations
- **NumPy/SciPy**: Scientific computing for image array operations
- **Sentence Transformers**: For generating text embeddings of radiology reports

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- Docker (for Qdrant)
- Git (for cloning repository)

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/dicom-viewer.git
   cd dicom-viewer
   
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Launch the application**:
   ```bash
   python app.py

## Requirements

This project uses Qdrant for vector database operations. Before running the application, ensure the Qdrant server is running. For Qdrant installation instructions, <a href="https://qdrant.tech/documentation/install/" target="_blank">you can click here</a>.
The setup above requires Docker. If you cannot use Docker, you can also run the Qdrant server using the <a href="https://qdrant.tech/documentation/install/#binary" target="_blank">here</a>
