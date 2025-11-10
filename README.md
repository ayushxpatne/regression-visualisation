## Regression Visualizer: Interactive Penalty & Gradient Descent Tool

This project is a sophisticated, interactive web application designed to provide a deep, practical understanding of linear and non-linear regression techniques, focusing specifically on different cost functions and the impact of hyperparameters.

It serves as an excellent educational tool for students to visualize abstract concepts like L1 vs. L2 penalties and the mechanics of Gradient Descent.

-----
## Deployment Status

### You can check out a live deployment of this tool here: **[https://regression-visualisation.onrender.com/](https://regression-visualisation.onrender.com/)**

>### **Note on Performance:** This application is hosted on a free tier service (Render) which may be slow to respond initially or during use. For the best experience and optimal speed, it is **highly recommended** to run the tool locally following the setup instructions below.

---

## Key Features

  * **Dual Regression Modes:** Supports both classic linear regression and advanced non-linear fitting (polynomial and sine wave).
  * **L1 Regression Visualization:** For linear models, users can manually adjust the slope ($m$) and intercept ($c$) to see the direct, real-time calculation of the **L1 Penalty** (Least Absolute Deviations). This demonstrates L1's robustness to outliers.
  * **Gradient Descent Simulation:** For polynomial and sine models, the tool uses the Gradient Descent algorithm with an **L2 Cost Function** (Least Squares Error) to find the best-fit curve.
  * **Hyperparameter Tuning:** Provides granular control over the key machine learning parameters for the Gradient Descent process:
      * **Order:** The degree of the polynomial to be fit.
      * **Learning Rate ($\mathbf{10^{-x}}$):** Control the step size of the optimizer using a logarithmic scale, ranging from $10^{0}$ to $10^{-10}$.
      * **Training Rounds:** Control the number of iterations for the optimization process (up to 10,000 rounds).

-----

## Technical Stack

This application is built using a minimal web stack, making it easy to deploy and understand:

  * **Backend:** Python (Flask)
  * **Scientific Computing:** NumPy for array manipulation and calculation.
  * **Visualization:** Matplotlib for generating the regression plots.
  * **Frontend:** HTML, JavaScript, and Tailwind CSS (or similar utility-first framework).

-----

## Setup and Installation

To run this application locally, you will need Python installed.

### 1\. Clone the Repository

```bash
git clone https://github.com/ayushxpatne/regression-visualisation.git
cd regression-visualisation
```

### 2\. Create and Activate a Virtual Environment

It is recommended to use a virtual environment for dependency management.

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 3\. Install Dependencies

Install the required Python packages (assuming `Flask`, `numpy`, and `matplotlib` are necessary).

```bash
pip install flask numpy matplotlib
```

### 4\. Run the Application

```bash
python app.py
```

The application will now be running on `http://127.0.0.1:5000`.
---

## Attribution and Development Notes

This tool was designed to explore regression penalties and optimize the core logic.

**Transparency Note:** The frontend code (HTML/CSS/Flask structure and dynamic JavaScript functions for the user interface) was developed and refined with the assistance of Claude 4.5 during the iterative building process.

If you find this tool helpful for teaching or learning, please consider starring the repository.
