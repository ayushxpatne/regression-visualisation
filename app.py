"""
Regression Visualizer - Educational Tool
Flask application for visualizing L1 regression
"""

from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import math

app = Flask(__name__)

# Store session data (in production, use sessions or database)
current_data = {}


def get_y_linear(x: int, m: int) -> int:
    """Generate y value with random variation"""
    c = random.randrange(0 - m * 3, m * 3)  # variation
    y = m * x + c
    return y


def get_l1_regression(x_arr, y_arr, guessed_m, guessed_c):
    """
    Calculate L1 regression penalty
    In linear equation the penalty using L1 regression is calculated as:
    sum of absolute values of |y - (mx + c)| where x and y = points
    """
    sum_val = 0
    err_values = []
    for i in range(len(x_arr)):
        err = abs(y_arr[i] - (guessed_m * x_arr[i] + guessed_c))
        err_values.append(err)
        sum_val += err

    return sum_val, err_values


def generate_plot(x_arr, y_arr, guessed_m, guessed_c, n):
    """Generate matplotlib plot and return as base64 string"""
    # Calculate guessed line data
    y_data_for_guessed_eqn = [guessed_m * x + guessed_c for x in x_arr]

    # Calculate penalty (L1 for linear)
    penalty, err_values = get_l1_regression(x_arr, y_arr, guessed_m, guessed_c)

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Plot actual data points
    ax.scatter(x_arr, y_arr, color="black", s=50, label="Actual Points", zorder=3)

    # Plot regression line
    x = np.linspace(0, n, 100)
    guessed_eqn = guessed_m * x + guessed_c
    ax.plot(
        x,
        guessed_eqn,
        color="black",
        linewidth=2,
        label=f"y = {guessed_m}x + {guessed_c}",
    )

    # Plot predicted points on line
    ax.scatter(
        x_arr,
        y_data_for_guessed_eqn,
        color="gray",
        s=30,
        label="Predicted Points",
        zorder=2,
    )

    # Draw error lines
    for i in range(len(y_arr)):
        ax.plot(
            [x_arr[i], x_arr[i]],
            [y_arr[i], y_data_for_guessed_eqn[i]],
            linestyle=":",
            color="gray",
            linewidth=1,
        )

    ax.set_xlabel("x", fontsize=12, family="monospace")
    ax.set_ylabel("y", fontsize=12, family="monospace")
    ax.set_title(
        f"L1 Regression Penalty: {penalty:.2f}", fontsize=14, family="monospace", pad=20
    )
    ax.legend(loc="upper left", frameon=False, prop={"family": "monospace"})
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_base64, penalty


@app.route("/")
def index():
    """Home page - generate points"""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate random points for regression"""
    n = int(request.form.get("n", 20))
    regression_type = request.form.get("regression_type", "linear")

    # Generate x values based on regression type
    if regression_type == "sine":
        # Sine wave data points are often between 0 and 1
        x_arr = [i / n for i in range(n)]
    else:
        x_arr = [random.randrange(0, n) for i in range(n)]

    # Generate y values based on regression type
    if regression_type == "linear":
        reg_m = int(request.form.get("reg_m", 3))
        y_arr = [get_y_linear(i, reg_m) for i in x_arr]
        current_data["true_m"] = reg_m
        current_data["regression_type"] = "linear"

    elif regression_type == "polynomial":
        poly_order = int(request.form.get("poly_order", 3))
        # Use a consistent max value for polynomial x to keep noise levels consistent
        x_arr = [random.randrange(0, 10) for i in range(n)]
        base_coeffs = [random.uniform(-2, 2) for _ in range(poly_order + 1)]
        y_arr = [get_y_polynomial(x, poly_order, base_coeffs) for x in x_arr]
        current_data["poly_order"] = poly_order
        current_data["base_coeffs"] = base_coeffs
        current_data["regression_type"] = "polynomial"

    elif regression_type == "sine":
        noise = float(request.form.get("noise", 1.0))
        y_arr = [get_y_sine(x, noise) for x in x_arr]
        current_data["noise"] = noise
        current_data["regression_type"] = "sine"

    # Store data
    current_data["x_arr"] = x_arr
    current_data["y_arr"] = y_arr
    current_data["n"] = n

    return jsonify(
        {
            "success": True,
            "x_arr": x_arr,
            "y_arr": y_arr,
            "n": n,
            "regression_type": regression_type,
        }
    )


@app.route("/visualize")
def visualize():
    """Visualization page - adjust m and c"""
    if not current_data:
        return "No data generated. Please go back and generate points first.", 400

    return render_template(
        "visualize.html", n=current_data["n"], true_m=current_data.get("true_m", 3)
    )


@app.route("/get_regression_type")
def get_regression_type():
    """Return current regression type"""
    return jsonify({"regression_type": current_data.get("regression_type", "linear")})


@app.route("/update_plot", methods=["POST"])
def update_plot():
    """Update plot with new parameters"""
    if not current_data:
        return jsonify({"error": "No data available"}), 400

    regression_type = request.json.get("regression_type", "linear")
    x_arr = current_data["x_arr"]
    y_arr = current_data["y_arr"]
    n = current_data["n"]

    if regression_type == "linear":
        guessed_m = float(request.json.get("m", 3))
        guessed_c = float(request.json.get("c", 0))
        img_base64, penalty = generate_plot(x_arr, y_arr, guessed_m, guessed_c, n)
        equation = f"y = {guessed_m}x + {guessed_c}"

    elif regression_type in ["polynomial", "sine"]:
        order = int(request.json.get("order", 3))
        # Learning rate is now passed directly as a float (e.g., 0.001)
        lr = float(request.json.get("learning_rate", 0.001))
        rounds = int(request.json.get("rounds", 1000))

        # Polynomial regression function is used for both polynomial and sine fitting
        # It performs an L2 (squared error) minimization with normalization
        penalty, weights, x_min, x_max = get_polynomial_regression(
            x_arr, y_arr, order, lr, rounds
        )

        # Generate predictions using normalized x
        y_pred = []
        for x in x_arr:
            x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else x
            # The weights are for the polynomial: w[0]*x^order + w[1]*x^(order-1) + ...
            pred = sum(
                weights[j] * (x_norm ** (order - j)) for j in range(len(weights))
            )
            y_pred.append(pred)

        img_base64 = generate_nonlinear_plot(
            x_arr, y_arr, y_pred, n, regression_type, order, penalty
        )
        equation = (
            f"{regression_type} regression (order={order})"
            if regression_type == "polynomial"
            else f"Fit sin(2πx) with order {order}"
        )

    return jsonify({"image": img_base64, "penalty": penalty, "equation": equation})


# The rest of the functions (generate_nonlinear_plot, get_y_polynomial, get_y_sine, get_polynomial_regression) remain unchanged as they support the required functionality.
def generate_nonlinear_plot(x_arr, y_arr, y_pred, n, regression_type, order, penalty):
    """Generate matplotlib plot for non-linear regression"""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Set fixed limits for sine wave
    if regression_type == "sine":
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, 1.5)

        # Add true sine function
        x_true = np.linspace(0, 1, 200)
        y_true = np.sin(2 * np.pi * x_true)
        ax.plot(
            x_true,
            y_true,
            linestyle="dashed",
            color="#ADD8E6",
            linewidth=2,
            label="True Function",
            alpha=0.7,
            zorder=1,
        )

    # Plot actual data points
    ax.scatter(x_arr, y_arr, color="black", s=50, label="Actual Points", zorder=3)

    # Sort for smooth line plotting
    sorted_pairs = sorted(zip(x_arr, y_pred))
    x_sorted, y_sorted = zip(*sorted_pairs)

    # Plot regression line/curve
    ax.plot(
        x_sorted,
        y_sorted,
        color="red",
        linewidth=2,
        label=f"{regression_type.capitalize()} Fit",
        zorder=2,
    )

    # Plot predicted points
    ax.scatter(
        x_arr, y_pred, color="gray", s=30, label="Predicted Points", zorder=2, alpha=0.6
    )

    # Draw error lines
    for i in range(len(y_arr)):
        ax.plot(
            [x_arr[i], x_arr[i]],
            [y_arr[i], y_pred[i]],
            linestyle=":",
            color="gray",
            linewidth=1,
            alpha=0.5,
        )

    ax.set_xlabel("x", fontsize=12, family="monospace")
    ax.set_ylabel("y", fontsize=12, family="monospace")

    title = f"L2 Error: {penalty:.2f} | Order: {order}"
    if regression_type == "sine":
        title = (
            f"L2 Error: {penalty:.2f} | Fitting sin(2πx) with Order {order} Polynomial"
        )

    ax.set_title(title, fontsize=14, family="monospace", pad=20)
    ax.legend(loc="upper left", frameon=False, prop={"family": "monospace", "size": 9})
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_base64


def get_y_polynomial(x: float, order: int, base_coeffs: list = None) -> float:
    """Generate y value for polynomial with random variation"""
    if base_coeffs is None:
        base_coeffs = [random.uniform(-2, 2) for _ in range(order + 1)]

    y = sum(coeff * (x ** (order - i)) for i, coeff in enumerate(base_coeffs))
    noise = random.uniform(-0.5, 0.5)
    return y + noise


def get_y_sine(x: float, noise_level: float) -> float:
    """Generate y value for sine wave with random variation"""
    c = random.uniform(-noise_level, noise_level)
    y = math.sin(2 * math.pi * x + c / 10)
    return y


def get_polynomial_regression(x_arr, y_arr, order, learning_rate=0.01, max_rounds=1000):
    """Calculate polynomial regression using gradient descent with normalization"""
    import numpy as np

    # Normalize x values to [0, 1] range for better numerical stability
    x_min, x_max = min(x_arr), max(x_arr)
    x_normalized = [
        (x - x_min) / (x_max - x_min) if x_max != x_min else x for x in x_arr
    ]

    # Initialize weights randomly (smaller values for stability)
    w = [random.uniform(-0.5, 0.5) for _ in range(order + 1)]

    best_error = float("inf")
    best_weights = w.copy()
    patience = 50
    no_improvement = 0

    for round_num in range(max_rounds):
        # Calculate predictions using normalized x
        y_pred = []
        for x in x_normalized:
            pred = sum(w[j] * (x ** (order - j)) for j in range(len(w)))
            y_pred.append(pred)

        # Calculate errors
        errors = [y_pred[i] - y_arr[i] for i in range(len(y_arr))]
        # L2 Error (Mean Squared Error scaled by 0.5)
        error = 0.5 * sum(e**2 for e in errors)

        if error < best_error:
            best_error = error
            best_weights = w.copy()
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stopping if no improvement
        if no_improvement > patience:
            break

        # Check for divergence
        if error > 1e6 or any(abs(weight) > 1e3 for weight in w):
            w = best_weights.copy()
            break

        # Calculate gradients with L2 regularization
        grads = [0] * len(w)
        lambda_reg = 0.001  # Regularization parameter

        for j in range(len(w)):
            grad_sum = 0
            for i in range(len(x_normalized)):
                power = order - j
                grad_sum += errors[i] * (x_normalized[i] ** power)
            # Add regularization term
            grads[j] = grad_sum / len(x_arr) + lambda_reg * w[j]

        # Adaptive learning rate
        adaptive_lr = learning_rate / (1 + round_num / 1000)

        # Update weights with gradient clipping
        for j in range(len(w)):
            grad_clipped = max(-10, min(10, grads[j]))
            w[j] -= adaptive_lr * grad_clipped

    # Final penalty is the L2 error of the best weights
    final_y_pred = []
    for x in x_normalized:
        pred = sum(
            best_weights[j] * (x ** (order - j)) for j in range(len(best_weights))
        )
        final_y_pred.append(pred)
    final_errors = [final_y_pred[i] - y_arr[i] for i in range(len(y_arr))]
    final_error = 0.5 * sum(e**2 for e in final_errors)

    # Note: The original front-end was set up to display L1 (sum of absolute errors),
    # but the get_polynomial_regression function is minimizing L2 (sum of squared errors).
    # Since the front-end will display a penalty, we will return the L2 error for this path.
    return final_error, best_weights, x_min, x_max


if __name__ == "__main__":
    app.run(debug=True)
