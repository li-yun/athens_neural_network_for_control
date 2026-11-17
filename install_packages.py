"""
Install and check all packages needed for the course exercises.

Run it with the SAME Python that your Jupyter kernel uses:
    - terminal:          python install_packages.py
    - or in a notebook:  %run install_packages.py
    - Google Colab:      upload the course files first (see below), then  %run install_packages.py

Google Colab: most packages are already installed. Upload install_packages.py,
requirements.txt, building_model.py and statespace.csv with the folder icon on the
left. Files uploaded this way are deleted when the runtime is reset.

The script
    1. installs the packages from requirements.txt into this Python,
    2. checks that every package can be imported,
    3. runs a small test of each tool used in the exercises.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / "requirements.txt"
IN_COLAB = "COLAB_RELEASE_TAG" in __import__("os").environ or "google.colab" in sys.modules


def header(text):
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


def install():
    header(f"1/3  Installing packages into:\n     {sys.executable}")
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}" + ("   (Google Colab)" if IN_COLAB else ""))
    if v < (3, 9):
        sys.exit("Python 3.9 or newer is required. Please install a newer Python.")
    # make sure pip itself is available in this Python
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        print("pip not found -> installing it with ensurepip")
        if subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"]).returncode != 0:
            sys.exit("[!] Could not install pip. Please install pip for this Python and try again.")
    # no --upgrade: only missing or too old packages are installed, existing ones are kept
    # (important on Colab, where upgrading pre-installed packages can break the runtime)
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    print(" ".join(cmd), "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[!] pip reported an error (see above).")
        if IN_COLAB:
            print("    Try: Runtime > Disconnect and delete runtime, then run this script again.")
        else:
            print("    Most often a package has no ready-made wheel for a very new Python version.")
            print("    Use Python 3.10 - 3.13, e.g. with Anaconda / Miniconda, and run this script again.")
        sys.exit(1)


def check_imports():
    header("2/3  Checking imports")
    import importlib
    ok = True
    for module in ["numpy", "pandas", "scipy", "matplotlib", "cvxpy", "sklearn"]:
        try:
            m = importlib.import_module(module)
            print(f"  [ok] {module:<11} {getattr(m, '__version__', '')}")
        except Exception as err:  # noqa: BLE001
            print(f"  [!!] {module:<11} {err}")
            ok = False
    return ok


def smoke_tests():
    header("3/3  Testing the tools used in the exercises")
    import numpy as np
    ok = True

    # cvxpy + CLARABEL (Exercises 2 and 4)
    try:
        import cvxpy as cp
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0, x <= 0.5])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL and np.allclose(x.value, 0.5, atol=1e-4)
        print("  [ok] cvxpy solves a QP with CLARABEL")
    except Exception as err:  # noqa: BLE001
        print(f"  [!!] cvxpy / CLARABEL: {err}")
        ok = False

    # scikit-learn neural network (Exercises 3 and 4)
    try:
        from sklearn.neural_network import MLPRegressor
        X = np.linspace(-1, 1, 50).reshape(-1, 1)
        MLPRegressor(hidden_layer_sizes=(4,), solver="lbfgs", max_iter=500).fit(X, X.ravel())
        print("  [ok] scikit-learn trains an MLPRegressor")
    except Exception as err:  # noqa: BLE001
        print(f"  [!!] scikit-learn: {err}")
        ok = False

    # course model and data (all exercises)
    try:
        sys.path.insert(0, str(HERE))
        from building_model import FirstOrderBuilding, outdoor_temperature
        FirstOrderBuilding().simulate(20.0, 0.0, outdoor_temperature("winter"))
        print("  [ok] building_model.py and statespace.csv are found")
    except Exception as err:  # noqa: BLE001
        print(f"  [!!] building_model: {err}")
        if IN_COLAB:
            print("       Upload building_model.py and statespace.csv (folder icon on the left) and run again.")
        else:
            print("       Keep install_packages.py, building_model.py and statespace.csv in the same folder.")
        ok = False

    # matplotlib
    try:
        from matplotlib.figure import Figure   # no GUI backend needed
        Figure().subplots().plot([0, 1])
        print("  [ok] matplotlib can plot")
    except Exception as err:  # noqa: BLE001
        print(f"  [!!] matplotlib: {err}")
        ok = False
    return ok


if __name__ == "__main__":
    install()
    if check_imports() and smoke_tests():
        restart = ("Runtime > Restart session" if IN_COLAB else "restart the kernel")
        header("All set! You can open the exercise notebooks.\n"
               f"(If packages were installed from a notebook, {restart} once.)")
    else:
        header("Some checks failed - see the messages marked [!!] above.")
        sys.exit(1)
