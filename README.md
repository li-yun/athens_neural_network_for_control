# ATHENS course — control of indoor temperature

Exercises on a first-order building thermal model (1R1C):

| | Notebook | Google Colab |
|---|---|---|
| Exercise 1 | `ex1_P_control.ipynb` | [Open in Colab](https://colab.research.google.com/github/li-yun/athens_neural_network_for_control/blob/main/ex1_P_control.ipynb) |
| Exercise 2 | `ex2_MPC.ipynb` | [Open in Colab](https://colab.research.google.com/github/li-yun/athens_neural_network_for_control/blob/main/ex2_MPC.ipynb) |
| Exercise 3 | `ex3_NN_MPC.ipynb` | [Open in Colab](https://colab.research.google.com/github/li-yun/athens_neural_network_for_control/blob/main/ex3_NN_MPC.ipynb) |
| Exercise 4 | `ex4_NN_approx_MPC.ipynb` | [Open in Colab](https://colab.research.google.com/github/li-yun/athens_neural_network_for_control/blob/main/ex4_NN_approx_MPC.ipynb) |

1. Saturated P control
2. Model predictive control (cvxpy)
3. Neural-network model and MPC by enumeration
4. Approximating MPC with a neural network

## Option A: Google Colab (nothing to install)

Click an *Open in Colab* link and run the first code cell. It downloads the course files and checks the packages.
If it says packages were installed, use *Runtime > Restart session* and continue.
To keep your work, use *File > Save a copy in Drive*.

## Option B: your own computer

1. Download and unzip the course folder (OneDrive link from your teacher), or
   `git clone https://github.com/li-yun/athens_neural_network_for_control.git`
2. Use **Python 3.10 – 3.13**, e.g. Anaconda/Miniconda.
3. In the course folder, run once:

       python install_packages.py

   You should see **All set!** at the end.
4. Open the notebooks in Jupyter or VS Code. Keep all files in the same folder.

## Files

- `building_model.py`: the building model and helper functions used by all exercises
- `statespace.csv`: measured data (outdoor temperature profile)
- `install_packages.py`, `requirements.txt`: installation and check of the Python packages
