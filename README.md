# UIDAI Hackathon - Aadhaar Data Analysis

**BY - Team ABHEDYA**

**Our ML-Model = [https://github.com/ayushrewd/ARIMA-ML-MODEL-__-TEAM-ABHEDYA/tree/main/uidai_ml](https://github.com/ayushrewd/ARIMA-ML-MODEL-__-TEAM-ABHEDYA)**

## Overview
This project analyzes Aadhaar enrolment, demographic updates, and biometric updates data to provide insights for UIDAI's service optimization and policy decisions.

## Features
- **State-wise Analysis**: Enrolment trends across Indian states
- **Age Group Analysis**: Distribution across different age segments
- **District-level Insights**: Top performing districts
- **Demographic Updates**: Migration and address change patterns
- **Biometric Updates**: Authentication stress indicators
- **Anomaly Detection**: Statistical outliers for audit
- **Time Series Analysis**: Daily trends and patterns
- **State Clustering**: ML-based state grouping

## Setup

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
Place your CSV files in the appropriate folders:
- `data/enrolment/` - Aadhaar enrolment data
- `data/demographic/` - Demographic update data
- `data/biometric/` - Biometric update data

## Usage
```bash
python aadhaar_analysis.py
```

## Output
All analysis results are saved to:
- `outputs/charts/` - Visualizations (PNG files)
- `outputs/tables/` - Data summaries (CSV files)

## Key Insights
- Identifies high-demand states and districts
- Detects age-group gaps for targeted campaigns
- Reveals migration patterns through demographic updates
- Highlights authentication challenges via biometric data
- Provides data-driven recommendations for UIDAI

## Project Structure
```
UIDAI/
├── data/
│   ├── enrolment/
│   ├── demographic/
│   └── biometric/
├── outputs/
│   ├── charts/
│   └── tables/
├── aadhaar_analysis.py
├── requirements.txt
└── README.md
```
