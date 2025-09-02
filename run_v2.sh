#!/bin/bash

# Activate the correct virtual environment
source ~/envs/ppb_env/bin/activate

# Confirm Python environment
echo "Using Python:"
which python
python --version
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import torch; print('Torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# Run the training script
python clinicalbert_train_with_symptom_dict.py
