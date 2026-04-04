#!/bin/bash

# Setup script for test_runner module
# Usage: bash test_runner/setup.sh

set -e

echo "================================"
echo "LangCoop Test Runner Setup"
echo "================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python: $PYTHON_VERSION"

# Check if Python 3.12+
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 12 ]); then
    echo "⚠ Warning: Python 3.12+ recommended. You have $PYTHON_VERSION"
fi

echo ""
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv_test" ]; then
    python3 -m venv venv_test
    echo "✓ Virtual environment created: venv_test/"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Step 2: Activating virtual environment..."
source venv_test/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "Step 3: Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r test_runner/requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "Step 4: Verifying installation..."
python3 -c "import carla; print(f'✓ CARLA version: {carla.__version__}')"
python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python3 -c "import yaml; print('✓ PyYAML installed')"
python3 -c "import cv2; print(f'✓ OpenCV version: {cv2.__version__}')"
python3 -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"

echo ""
echo "================================"
echo "Setup Complete! 🎉"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Start CARLA server:"
echo "   ./CARLA_0.9.16/CarlaUE4.sh -quality-level=Low"
echo ""
echo "2. In another terminal, activate venv and run test:"
echo "   source venv_test/bin/activate"
echo "   python test_runner_example.py --steps 500"
echo ""
echo "For more options:"
echo "   python test_runner_example.py --help"
echo ""
