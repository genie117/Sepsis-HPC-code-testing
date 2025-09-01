#!/bin/bash

# Activate virtual environment
source ~/myenv/bin/activate

# Confirm Python version and transformers version
echo "Using Python:"
which python
python --version
python -c "import transformers; print('Transformers version:', transformers.__version__)"

# Run the training script
python clinicalbert_train_with_symptom_dict.py

# Deactivate the environment
deactivate
