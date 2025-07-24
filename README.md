# Toward Automated AR Testing: Playback-Driven and LLM-Assisted Real-World Framework

[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/ARTesting/Playback_AR_Testing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

> **An automated testing framework for mobile AR applications that leverages pre-recorded videos, automated UI testing, and Large Language Models for comprehensive quality assessment.**

## 🎯 Overview

This repository contains the implementation of our automated AR testing framework presented in **ASE 2025**. Our approach addresses the critical challenges in AR app testing by:

- **🎬 Playback-Driven Testing**: Using pre-recorded real-world scenarios to eliminate manual scene setups
- **🤖 LLM-Assisted Evaluation**: Leveraging state-of-the-art vision-language models for quality assessment  
- **📊 Comprehensive Metrics**: Evaluating 6 key AR quality dimensions with high precision
- **⚡ Open-Source & Accessible**: Cost-effective solution using open-source models

## 🚀 Key Features

### 📹 Reusable Test Scenarios
- **36 pre-recorded playback videos** covering diverse environments (indoor/outdoor, various lighting conditions)
- Cross-device compatibility using ARCore's Recording & Playback API
- Eliminates need for repeated physical scene setups

### 🎯 Automated Quality Assessment
- **6 AR Quality Metrics**: Object Placement, Object Movement, Occlusion, Lighting, Visual Artifacts, Black Screen
- **83.3% precision improvement** over human annotation using LLM majority voting
- **76.68% average accuracy** with open-source vision-language models

### 🛠️ Complete Testing Pipeline
- Automated app instrumentation for playback support
- Multi-threaded testing framework with realistic user interactions
- JSON-formatted results for easy integration

## 📊 Performance Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Qwen2.5-VL-72B-Instruct | **81.48%** | **63.42%** | 33.22% | 43.60% |
| Mistral-Small-3.1-24B | 80.28% | 54.64% | 50.18% | **52.31%** |
| Gemma-3-27b-it | 74.73% | 42.07% | 45.69% | 43.81% |
| InternVL2.5-26B-MPO | 70.22% | 37.48% | 57.59% | 45.48% |

*Performance against LLM-generated ground truth labels*


## 🔧 Installation

### Prerequisites
- Python 3.8+
- Android Debug Bridge (ADB)
- Android device with ARCore support
- CUDA-compatible GPU (recommended)


## 🚀 Quick Start

### 1. Try the Online Demo
Visit our [Hugging Face Space](https://huggingface.co/spaces/ARTesting/Playback_AR_Testing) to test AR videos directly in your browser!


## 📋 AR Quality Metrics

Our framework evaluates AR applications across six critical dimensions:

| Metric | Description | Examples |
|--------|-------------|----------|
| **Object Placement** | Spatial accuracy relative to real environment | Floating objects, incorrect surface alignment |
| **Object Movement** | Stability and responsiveness during interaction | Jittery motion, tracking drift |
| **Occlusion** | Correct depth ordering with real objects | Objects appearing through walls |
| **Lighting** | Integration with environmental lighting | Missing shadows, incorrect reflections |
| **Visual Artifacts** | Rendering quality issues | Aliasing, texture problems, flickering |
| **Black Screen** | System failures and crashes | App freezes, rendering failures |

## 📁 Dataset

### Playback Videos
We provide **36 carefully designed test scenarios** covering:
- **Environments**: Indoor/outdoor settings
- **Object Sizes**: Small (<1.5m), medium (1.5-2.5m), large (>2.5m)
- **Scene Complexity**: Single plane, multi-plane, complex environments
- **Lighting Conditions**: Bright, moderate, shaded, dim

### Tested Applications
Our evaluation includes **15 real-world AR apps** with **880 test recordings**.

## 🎯 Ground Truth Generation

We establish reliable ground truth through **majority voting** among three commercial LLMs:
- **OpenAI o3**
- **Claude-3.7-Sonnet (Extended Thinking)**  
- **Gemini-2.5-Pro**

This approach achieves:
- **83.3% precision improvement** over human annotation
- **40.5% F1-score improvement**
- **77.31% unanimous agreement** rate among models

## 📊 Evaluation Results

### Model Performance Comparison
Our framework demonstrates that open-source models can achieve competitive performance for AR quality assessment:

- **Best Overall Accuracy**: Qwen2.5-VL-72B-Instruct (81.48%)
- **Best Balanced Performance**: Mistral-Small-3.1-24B (F1: 52.31%)
- **Cost-Effective Option**: Qwen2.5-VL-3B-Instruct (fast inference, good quality)

### Identified Issues
Across 880 test recordings, our framework successfully identified:
- **Temporal displacement errors** undetectable by static analysis
- **Lighting integration problems** in various environmental conditions
- **Object placement issues** across different surface types
- **Rendering artifacts** and system failures

## 🔬 Research Applications

This framework enables researchers and developers to:

### For Researchers
- **Reproducible AR Testing**: Standardized evaluation methodology
- **Benchmarking**: Compare AR apps and algorithms systematically  
- **Dataset Creation**: Generate labeled datasets for AR quality research

### For Developers
- **Automated QA**: Integrate into CI/CD pipelines
- **Issue Detection**: Identify problems before user deployment
- **Performance Optimization**: Targeted improvements based on specific metrics

### For Industry
- **Cost Reduction**: Minimize manual testing effort
- **Quality Assurance**: Consistent, objective evaluation
- **User Experience**: Improve AR app quality systematically


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ARCore Team** for the Recording & Playback API
- **Hugging Face** for model hosting and deployment platform
- **Open-source VLM communities** for making advanced models accessible
- **Research participants** who contributed to evaluation and validation


**⭐ Star this repository if you find it useful for your AR development or research!**