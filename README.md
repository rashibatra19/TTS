# Text-to-Speech (TTS) Fine-tuning Project

## Table of Contents
- [Overview](#overview)
- [Introduction](#introduction)
- [Technical Implementation](#technical-implementation)
- [Methodology](#methodology)
  - [English Technical Speech](#english-technical-speech)
  - [Hindi Language](#hindi-language)
- [Results and Evaluation](#results-and-evaluation)
- [Challenges and Future Work](#challenges-and-future-work)

## Overview

This project focuses on fine-tuning text-to-speech (TTS) models for two specific use cases:
1. Technical vocabulary in English (specialized for technical interviews)
2. Hindi language support

The implementation uses Coqui TTS as the base model and employs transfer learning techniques for optimization.

## Introduction

### What is Text-to-Speech?
Text-to-Speech (TTS) is a technology that converts written text into spoken words by combining NLP and speech synthesis algorithms. Modern TTS systems use neural networks to generate natural-sounding speech with appropriate prosody and emotion.

### Key Components
- Text Preprocessing
- Grapheme-to-Phoneme (G2P) Conversion
- Prosody Generation
- Waveform Generation

### Applications
- Accessibility tools
- Voice assistants
- Educational technology
- Customer service
- Content creation
- Assistive technology for elderly and disabled

## Technical Implementation

### Model Selection: Coqui TTS
Selected for its:
- Flexibility and multi-speaker capabilities
- Precise phoneme control
- Strong community support
- Open-source nature

### Environment Setup

```python
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
```

### Base Configuration

```python
RUN_NAME = "xtts_fine_tuning"
PROJECT_NAME = "tts_project"
DASHBOARD_LOGGER = "wandb"
BATCH_SIZE = 1
GRAD_ACUMM_STEPS = 252
```

## Methodology

### English Technical Speech

#### Dataset Collection
1. Web Scraping Technical Content
```python
def extract_sentences(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(paragraph.text.split('. '))
        return sentences
    except Exception as e:
        print(f"Error while processing URL {url}: {e}")
        return []
```

#### Preprocessing Steps
1. Text Cleaning
```python
def clean_text(text):
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
```

2. Technical Term Handling
```python
def preprocess_for_tts(text):
    text = text.replace("API", "A P I")
    text = text.replace("SDK", "S D K")
    text = text.replace("TTS", "T T S")
    return text
```

### Hindi Language Support

#### Dataset Preparation
- Used VoxPopuli Hindi Dataset
- Modified to match LJSpeech format
- Special handling for Devanagari script

```python
def remove_spaces_in_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    cleaned_content = re.sub(r'\s*\|\s*', '|', content)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
```

#### Model Configuration for Hindi
```python
model_args = GPTArgs(
    max_conditioning_length=82720,
    min_conditioning_length=8480,
    max_wav_length=82720,
    max_text_length=1000,
    language="hi"
)
```

## Results and Evaluation

### Performance Metrics
- MOS Score: 3.9 out of 5
- Pronunciation Accuracy: High for technical terms
- Speech Clarity: Good intelligibility
- Real-time Performance: Low latency

### Key Findings
1. English Technical Terms
   - Accurate pronunciation of acronyms
   - Natural flow in technical contexts
   - Some challenges with complex sentences

2. Hindi Language
   - Good handling of Devanagari script
   - Natural-sounding pronunciation
   - Room for improvement in prosody

## Challenges and Future Work

### Challenges
1. Data Collection
   - Gathering diverse technical vocabulary
   - Balanced representation of terms
   - Format standardization

2. Resource Constraints
   - Limited GPU availability
   - Training time restrictions
   - Memory management

3. Technical Hurdles
   - Acronym pronunciation
   - Prosody control
   - Long sentence handling

### Future Improvements
1. Model Enhancement
   - Extended training epochs
   - Larger batch sizes
   - Advanced prosody control

2. Data Optimization
   - Expanded technical vocabulary
   - More diverse sentence structures
   - Better balance of terms

3. Deployment
   - API development
   - Web interface
   - Cloud deployment options

## Directory Structure
```
project/
├── data/
│   ├── english_technical/
│   │   ├── wavs/
│   │   └── metadata.txt
│   └── hindi/
│       ├── wavs/
│       └── transcripts.txt
├── models/
│   ├── checkpoints/
│   └── configs/
└── scripts/
    ├── preprocessing/
    ├── training/
    └── evaluation/
```

## Resources and Dependencies
- Coqui TTS
- Python 3.8+
- PyTorch
- BeautifulSoup4
- Librosa
- Wandb (for logging)

## Acknowledgments
Special thanks to:
- The Coqui TTS community
- VoxPopuli dataset creators
- Project contributors and evaluators
