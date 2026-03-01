# FX Valuation - Machine Learning for Fair Value Prediction
Using Machine Learning to determine short-term fair values for FX rates

## Project Overview

This project applies advanced machine learning techniques to predict short-term fair values for foreign exchange (FX) rates. The repository is structured for maximum reproducibility and follows industry best practices for data science projects.

### Key Features
- Modular code structure for easy maintenance and scaling
- Comprehensive data pipeline from raw data to model predictions
- Jupyter notebooks for exploratory data analysis and visualization
- Production-ready virtual environment setup
- Reproducible results with version-controlled dependencies

---

## Getting Started

### Prerequisites
- **Python 3.8 or higher** (Download from [python.org](https://www.python.org/))
- **Git** (for cloning the repository)
- **Windows PowerShell** (for running the setup script)

### Installation & Setup

Follow these steps to set up the project on your machine:

#### 1. Clone the Repository
You can clone the repository using your preferred protocol. Examples:

**HTTPS:**
```bash
git clone https://github.com/alberto2002mp-2/fxvaluation_d200_am3483.git
cd fxvaluation_d200_am3483
```

**GitHub CLI:**
```bash
gh repo clone alberto2002mp-2/fxvaluation_d200_am3483
cd fxvaluation_d200_am3483
```

Replace the URL with the one appropriate for your account if you forked the repo.

#### 2. Run the Setup Script (Windows)
Execute the PowerShell setup script to automatically create a virtual environment and install dependencies:

```powershell
# If you encounter an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run the setup script:
.\setup_env.ps1
```

The script will:
- ✓ Verify Python installation
- ✓ Create a virtual environment (`venv` folder)
- ✓ Activate the virtual environment
- ✓ Upgrade pip, setuptools, and wheel
- ✓ Install all required dependencies from `requirements.txt`

#### 3. Manual Setup (Alternative Method)
If you prefer to set up manually or use a non-Windows system:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
fxvaluation_d200_am3483/
├── data/              # Dataset storage (raw and processed data)
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned and transformed data
├── notebooks/         # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb  # Exploratory Data Analysis
│   └── 02_Modeling.ipynb  # Model development and evaluation
├── src/              # Python modules and utilities
│   ├── __init__.py
│   ├── data.py       # Data loading and preprocessing
│   ├── features.py   # Feature engineering
│   └── models.py     # Model definitions and training
├── setup_env.ps1     # Automated setup script (Windows)
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
├── LICENSE           # Project license
└── README.md         # This file
```

---

## Dependencies

The project uses the following key Python libraries:

- **pandas** (2.0.0+): Data manipulation and analysis
- **numpy** (1.24.0+): Numerical computing
- **scikit-learn** (1.3.0+): Machine learning algorithms
- **matplotlib** (3.7.0+): Data visualization
- **xgboost** (2.0.0+): Gradient boosting framework
- **lightgbm** (4.0.0+): Fast gradient boosting
- **jupyter** (1.0.0+): Interactive computing environment
- **ipykernel** (6.25.0+): IPython kernel for Jupyter

For a complete list, see [requirements.txt](requirements.txt).

---

## Quick Start

Once the environment is set up, you can start working immediately:

### Activate the Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Launch Jupyter Notebook
```bash
jupyter notebook notebooks/
```

### Deactivate the Virtual Environment
```bash
deactivate
```

---

## Troubleshooting

### Issue: "Python is not recognized"
**Solution**: Add Python to your system PATH or reinstall Python with "Add Python to PATH" option selected.

### Issue: "The execution policy prevents running scripts"
**Solution**: Run this command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Virtual environment won't activate
**Solution**: Delete the `venv` folder and run `setup_env.ps1` again.

### Issue: Module import errors in Jupyter
**Solution**: Ensure Jupyter is using the correct kernel:
```bash
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

---



## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.


---

**Last Updated**: March 2026  
**Environment**: Python 3.8+, Virtual Environment (venv)
