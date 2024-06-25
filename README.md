This repository contains the data and code to reproduce all plots in the book chapter "Towards a Better Understanding of Misfit Through Explainable AI Techniques" by Corine Boon, Erdinç Durak, and İlker Birbil from the Faculty of Economics and Business, University of Amsterdam. If you have any questions about the data or the code, please contact me at [e.durak@uva.nl](mailto:e.durak@uva.nl).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to enhance our understanding of misfit using explainable AI techniques. It includes the necessary data and code scripts to generate the plots presented in the book chapter. You can follow this README file to reproduce the plots or apply the models to your data on your PC, or you can follow [this notebook](https://colab.research.google.com/drive/15EdqUIpe-8kTYpuw4J06DpsPDv7eZGJl#scrollTo=VL-FNnfztn2_) to generate the plots on the browser. If you are not familiar with Python, we strogly recommend you work on the notebook.

## Installation

### Prerequisites

Ensure you have the following installed on your PC:

- Python (version 3.11)

### Steps

1. **Clone the Repository**

   Download all the files in this repository to your PC.

2. **Install Gurobi**

   Please install the Gurobi optimizer on your PC by visiting [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/). You can have a free license for academic purposes.

## Usage

### Reproduce the Results

To reproduce the plots, run the script `AllPlots.py` on the Python environment.

### Analyze Your Data

To analyze your own data, please follow these steps:

1. **Prepare Your Data**

   Put your data (named `Data.csv`) into the same file with the `AllPlots.py` with the following format:
   - The columns should be in the order of: firstly the characteristics of person (should be named as P-_characteristics1_, P-_characteristics2_, etc.), and then the characteristics of environment (should be named as E-_characteristics1_, E-_characteristics2_, etc., E can be changed to J, O, etc.). After the characteristics of person and environment, contextual variables can be added, and lastly the outcome should be added.
  - Be sure that the number of characteristics of person and environment are the same and the characteristics are in the same order.
  - Be sure that your data includes only numerical data.
  - Be sure that there is no missing data.
  - Be sure that your outcome is a binary variable.

2. **Update the Code**

   Follow the comments on the script `AllPlots.py`, and make the necessary changes under the comments starting with "Please". Save the changes.

3. **Run the Code**

   Run the script `AllPlots.py` on the Python environment

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
