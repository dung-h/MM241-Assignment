# Cutting Stock Problem Optimization

## Project Overview
This project provides a comprehensive solution for the Cutting Stock Problem using a Parallel Genetic Algorithm approach. The application includes a graphical user interface for easy interaction, parallel genetic algorithm implementation, and performance evaluation tools.

## Features
- Genetic Algorithm optimization for cutting stock problems
- Graphical User Interface (GUI) for easy input and visualization
- Detailed performance metrics and evaluation
- Problem instance generation
- Multiple optimization strategies

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cutting-stock-problem.git
cd cutting-stock-problem
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Components
- `core_ga.py`: Core Genetic Algorithm and Parallel Genetic Algorithm implementation
- `gui.py`: User interface for problem setup and visualization
- `metric.py`: Performance evaluation and analysis tools
- `2DCPackGen.exe`: Problem instance generator, reference this link for more detail: https://github.com/Oscar-Oliveira/OR-Datasets/tree/b15c160b80673f03604c71e86cea625e11713c2b/Cutting-and-Packing/2D/Generators/2DCPackGen

## How to Use

### 1. Launch the Application
```bash
python gui.py
```

### 2. Using the GUI
1. Load Stock Data:
   - Click "Select Stock File" to import stock dimensions
   - Supported format: CSV with columns [id, length, width]

2. Load Demand Data:
   - Click "Select Demand File" to import product demand
   - Supported format: CSV with columns [id, length, width, quantity]

3. Run Optimization:
   - Click "Run Optimization" to start genetic algorithm
   - View results in the visualization panel

4. Export Results:
   - Use "Export Results" to save optimization outcomes

# Note: Sample of stock.csv and demand.csv are given.
### 3. Performance Evaluation
```bash
python metric.py
```
- Generates comprehensive performance metrics
- Runs multiple optimization iterations
- Produces statistical analysis

## Customization
- Modify parallel genetic algorithm parameters in `core_ga.py`
- Adjust problem generation settings in `Format_input.txt` and rerun the `2DCPackGen.exe`

## Performance Metrics
- Total Stock Used
- Area Utilization
- Cutting Efficiency

## Datasets
- This folder contains datasets used by the interal evaluation.

## Compare with exist tool, Problem 1 - Problem 10
- Contain the result of our tool and CutLogic 2D using dataset of Problem 1 and 10.

## Troubleshooting
- Ensure all dependencies are installed
- Check input file formats
- Verify stock and demand data compatibility

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Authors
TN01/Team 18
