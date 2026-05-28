# Machine Learning Practice Repository

A hands-on repository for learning and experimenting with machine learning concepts, algorithms, and workflows using Python. This project serves as a personal playground for practicing data preprocessing, feature engineering, model training, evaluation, and deployment concepts using real-world and synthetic datasets.

---

## Goals

- Learn core machine learning fundamentals
- Practice implementing ML algorithms and workflows
- Explore data preprocessing and feature engineering
- Build and evaluate predictive models
- Experiment with different datasets and techniques
- Improve understanding of model optimization and performance tuning

---

## Topics Covered

### Data Processing
- Data Cleaning
- Missing Value Handling
- Feature Scaling
- Encoding Categorical Variables
- Feature Engineering

### Exploratory Data Analysis (EDA)
- Data Visualization
- Statistical Analysis
- Correlation Analysis
- Outlier Detection

### Machine Learning

#### Supervised Learning
- Regression
- Classification

#### Unsupervised Learning
- Clustering
- Dimensionality Reduction

### Model Development
- Model Training
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Performance Metrics

### Advanced Topics
- Deep Learning
- NLP Experiments
- Computer Vision
- Time Series Forecasting
- Reinforcement Learning

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch / TensorFlow
- marimo
- uv

---

## Repository Structure

```bash
machine_learning/
│
├── datasets/          # Training and sample datasets
├── notebooks/         # marimo notebooks and experiments
├── projects/          # End-to-end ML projects
├── models/            # Saved trained models
├── scripts/           # Reusable ML scripts and utilities
├── experiments/       # Experimental implementations
├── tests/             # Unit and integration tests
├── pyproject.toml     # Project configuration and dependencies
├── uv.lock            # Locked dependency versions
└── README.md
```

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/samueladole/machine_learning
cd machine_learning
```

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment

```bash
uv venv
```

### Activate Environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
uv sync
```

---

## Usage

### Run marimo Notebook

```bash
uv run marimo edit notebooks/example.py
```

### Run Python Scripts

```bash
uv run python scripts/train_model.py
```

---

## Learning Philosophy

This repository focuses on learning machine learning through practical implementation and experimentation. The goal is to understand not only how to use ML libraries, but also:

- How machine learning algorithms work internally
- When to choose specific models
- Trade-offs between approaches
- Common pitfalls such as overfitting and data leakage
- Real-world machine learning workflows and best practices

---

## Planned Additions

- End-to-end ML pipelines
- MLOps experiments
- Model deployment examples
- Distributed training
- Real-world datasets
- Kaggle competition solutions
- Custom ML algorithm implementations

---

## Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and experiment freely.

---

## License

This project is intended for educational and practice purposes.


