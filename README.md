# GPT-v1

## Overview

GPT-v1 is a simple implementation of a Transformer-based language model inspired by GPT-2. This project includes a custom implementation of a lightweight GPT model designed for text generation tasks. The model is trained on a subset of the BookCorpus dataset and can generate text based on various predefined use cases.

## Features

- Custom GPT architecture for text generation.
- Configurable text generation parameters such as temperature, top-k, and top-p sampling.
- Training pipeline with dataset loading, tokenization, and model checkpointing.
- Efficient fine-tuning with AdamW optimizer and linear warmup scheduler.
- Multiple text generation modes: code generation, chatbot responses, creative writing, and more.

## File Structure

```
.
├── decoder.py             # Decoder module for the GPT model
├── gpt.py                 # Main GPT model implementation
├── main.py                # Training and inference script
└── __pycache__/           # Cached Python files
```

## Installation

To set up the project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run the `main.py` script. The script will automatically load the BookCorpus dataset, tokenize the text, and start the training process.

```bash
python main.py
```

Checkpoints are saved periodically, allowing you to resume training from the last checkpoint.

### Text Generation
After training, the model can generate text based on different configurations. Edit the `use_case` variable in `main.py` to select a mode such as `creative_writing`, `chatbot`, or `code_generation`.

Example:

```python
use_case = "creative_writing"
```

Run the script in test mode to generate text:

inside `main.py` change this line to

```python
test_mode = True
```

Then run
```bash
python main.py
```

### Checkpointing
Training progress is saved in `last_checkpoint.pth`, which includes:
- Model state
- Optimizer state
- Scheduler state
- Training progress

If a checkpoint exists, training will resume automatically.

## Dependencies
- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Datasets
- TQDM

For a complete list, refer to `requirements.txt`.

## License
This project is released under the MIT License.

## Author
PelikanFix16

