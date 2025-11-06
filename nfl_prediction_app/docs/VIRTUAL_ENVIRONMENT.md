# Virtual Environment Setup Guide

## Why Use a Virtual Environment?

A virtual environment isolates your project's dependencies from your system's base Python installation. This prevents:
- Version conflicts between different projects
- Polluting your base Python installation
- Dependency issues when sharing code with others

## Step-by-Step Setup

### Option 1: Using venv (Built-in, Recommended)

#### 1. Navigate to the project directory
```bash
cd nfl_prediction_app
```

#### 2. Create the virtual environment
```bash
# On macOS/Linux:
python3 -m venv venv

# On Windows:
python -m venv venv
```

This creates a `venv/` folder with an isolated Python installation.

#### 3. Activate the virtual environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

You'll see `(venv)` in your terminal prompt when activated.

#### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 5-10 minutes as it downloads all ML libraries.

#### 5. Verify installation
```bash
python -c "import nfl_data_py; import pandas; import sklearn; print('✅ All packages installed!')"
```

### Option 2: Using conda (If you have Anaconda/Miniconda)

#### 1. Create environment
```bash
conda create -n nfl_prediction python=3.11
```

#### 2. Activate environment
```bash
conda activate nfl_prediction
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Daily Workflow

### Starting Work
```bash
# Navigate to project
cd nfl_prediction_app

# Activate environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### Running Code
```bash
# Your environment is now active, run any script:
python scripts/run_pipeline.py
python web_app/app.py
pytest tests/
```

### Stopping Work
```bash
# Deactivate the environment
deactivate
```

## Common Issues & Solutions

### Issue: "python3: command not found"
**Solution:** Try `python` instead of `python3`

### Issue: PowerShell script execution disabled (Windows)
**Solution:** Run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "venv is not recognized" (Windows)
**Solution:** Use full path:
```cmd
C:\path\to\nfl_prediction_app\venv\Scripts\activate.bat
```

### Issue: Packages installing to wrong location
**Solution:** Verify activation:
```bash
which python  # macOS/Linux - should show venv/bin/python
where python  # Windows - should show venv\Scripts\python.exe
```

## Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Python location
which python  # Should show path to venv/bin/python

# 2. Check pip location
which pip     # Should show path to venv/bin/pip

# 3. List installed packages
pip list

# 4. Run tests
pytest tests/ -v

# 5. Test imports
python -c "from models.trainer import ModelTrainer; print('✅ Success!')"
```

## IDE Setup

### VS Code
1. Open project folder: `File > Open Folder > nfl_prediction_app`
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose: `./venv/bin/python` (or `venv\Scripts\python.exe` on Windows)

### PyCharm
1. Open project
2. Go to: `File > Settings > Project > Python Interpreter`
3. Click gear icon → `Add`
4. Select `Existing environment`
5. Navigate to `venv/bin/python`

### Jupyter Notebook
```bash
# Install Jupyter in your venv
pip install jupyter

# Launch notebook
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Updating Dependencies

If you add new packages to `requirements.txt`:

```bash
# Activate environment
source venv/bin/activate

# Install new packages
pip install -r requirements.txt

# Or install specific package
pip install package-name
```

## Deleting the Virtual Environment

If you need to start fresh:

```bash
# Deactivate if active
deactivate

# Delete the folder
rm -rf venv/  # macOS/Linux
rmdir /s venv  # Windows

# Recreate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Best Practices

1. **Always activate** before running project code
2. **Never commit** the `venv/` folder to git (already in `.gitignore`)
3. **Update requirements.txt** when you add new packages:
   ```bash
   pip freeze > requirements.txt
   ```
4. **Use same Python version** across team (3.11 recommended)

## Quick Reference

| Action | Command |
|--------|---------|
| Create venv | `python3 -m venv venv` |
| Activate (Mac/Linux) | `source venv/bin/activate` |
| Activate (Windows) | `venv\Scripts\activate` |
| Install deps | `pip install -r requirements.txt` |
| Deactivate | `deactivate` |
| Check Python | `which python` |
| Run tests | `pytest tests/` |
| Start app | `python web_app/app.py` |

## Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
ODDS_API_KEY=your_actual_key_here
```

The `.env` file is in `.gitignore` so your keys stay private.

---

**Note:** The `venv/` directory can be large (300+ MB) because it includes ML libraries. This is normal!
