This repository contains the data and code to reproduce all plots in the book chapter "Towards a Better Understanding of Misfit Through Explainable AI Techniques" by Corine Boon, Erdinç Durak, and İlker Birbil from the Faculty of Economics and Business, University of Amsterdam.

If you have any questions about the data or the code, please contact me at [e.durak@uva.nl](mailto:e.durak@uva.nl).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to enhance our understanding of misfit using explainable AI techniques. It includes the necessary data and scripts to generate the plots presented in the book chapter.

## Installation

### Prerequisites

Ensure you have the following installed on your PC:

- Python (version 3.11)
- Visual Studio Code (optional, but strongly recommended for better development experience, including code editing, debugging, and extensions support)
- Python extension for Visual Studio Code (version 3.11)

For further information, please see the [Visual Studio Code download page](https://code.visualstudio.com/download).

### Steps

1. **Clone the Repository**

   Download all the files in this repository to your PC.

3. **Install Required Packages**

   Install the necessary Python packages by running the following command in your terminal:

   ```sh
   pip install numpy pandas scikit-learn shap dtreeviz matplotlib
   ```

   For more information on installing packages, see the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/).

4. **Create and Activate a Virtual Environment**

   Run the following commands one by one in your terminal:

   ```sh
   python3.8 -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
   python --version
   pip list
   pip install dice-ml
   cd /path/to/your/cloned/repository
   pip install -r requirements.txt
   pip install jupyter
   ```

5. **Visit Gurobi**

   For additional optimization capabilities, visit [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/).

6. **Deactivate the Environment**

   After setup, you can deactivate the virtual environment with:

   ```sh
   deactivate
   ```

## Usage

To reproduce the plots, run the script `AllPlots.py`:

```sh
python AllPlots.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
