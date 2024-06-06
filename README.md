Dear reader,

This repository contains the data and the code to reproduce all plots in the book chapter "Towards a Better Understanding of Misfit Through Explainable AI Techniques" by Corine Boon, Erdinç Durak, and İlker Birbil from the Faculty of Economics and Business, University of Amsterdam. Please do not hesitate to contact me at e.durak@uva.nl if you have any questions about the data or the code.

Please follow these steps to reproduce the plots:

1. Download all the files in this repository to your PCs.
2. If you do not have the tools to run Python on your PC, download Visual Studio Code and the Python extension. For further information please see https://code.visualstudio.com/download.
3. Install these packages to the environment you are working on dtreeviz, matplotlib, numpy, pandas, scikit-learn, shap. You can install these packages by running the following command in your terminal:
   pip install numpy pandas scikit-learn shap dtreeviz matplotlib
   For further information please see https://packaging.python.org/en/latest/tutorials/installing-packages/.
4. In your terminal, run the following lines one by one:
   python3.8 -m venv venv
   source venv/bin/activate
   python --version
   pip list
   pip install dice-ml
   cd /Users/tabearober/Documents/Counterfactuals/CE-OCL
   pip install -r requirements.txt
   pip install jupyter
   visit https://www.gurobi.com/products/gurobi-optimizer/
   deactivate
5. Run "AllPlots.py".

Kind regards,
Erdinç Durak
