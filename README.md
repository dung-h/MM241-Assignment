# Cutting Stock Problem Optimization

## Project Overview
This project provides a comprehensive solution for the Cutting Stock Problem using a Genetic Algorithm approach. The application includes a graphical user interface for easy interaction, genetic algorithm implementation, and performance evaluation tools.

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
- `core_ga.py`: Core Genetic Algorithm implementation
- `gui.py`: User interface for problem setup and visualization
- `metric.py`: Performance evaluation and analysis tools
- `2DCPackGen.cpp`: Problem instance generator

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

### 3. Performance Evaluation
```bash
python metric.py
```
- Generates comprehensive performance metrics
- Runs multiple optimization iterations
- Produces statistical analysis

## Customization
- Modify genetic algorithm parameters in `core_ga.py`
- Adjust problem generation settings in `2DCPackGen.cpp`

## Performance Metrics
- Total Stock Used
- Area Utilization
- Cutting Efficiency
- Computational Time

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

