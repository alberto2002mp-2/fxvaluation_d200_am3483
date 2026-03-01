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
```bash
git clone <repository-url>
cd fxvaluation_d200_am3483
```

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

## Working with the Project

### Adding New Dependencies
If you need to install additional packages:

1. Activate the virtual environment
2. Install the package: `pip install package-name`
3. Update requirements.txt: `pip freeze > requirements.txt`
4. Commit the updated `requirements.txt` to version control

### Creating New Notebooks
1. Create files in the `notebooks/` directory with descriptive names (e.g., `03_Feature_Engineering.ipynb`)
2. Number them in execution order for clarity

### Adding Python Modules
1. Create `.py` files in the `src/` directory
2. Import and use them in your notebooks as:
   ```python
   from src.data import load_data
   from src.models import train_model
   ```

---

## Best Practices

✓ **Always work within the virtual environment** to maintain isolation  
✓ **Keep raw data in `data/raw/`** and never modify it  
✓ **Document your analysis** in notebook markdown cells  
✓ **Use consistent naming conventions** for variables and functions  
✓ **Commit `requirements.txt` changes** when dependencies change  
✓ **Add meaningful commit messages** for reproducibility  

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

## Contributing

When collaborating on this project:

1. Create a new branch for your feature: `git checkout -b feature-name`
2. Make your changes and commit with clear messages
3. Push to the repository: `git push origin feature-name`
4. Create a Pull Request describing your changes

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Contact & Support

For questions or issues, please refer to the project documentation or contact the project maintainers.

---

**Last Updated**: March 2026  
**Environment**: Python 3.8+, Virtual Environment (venv)
