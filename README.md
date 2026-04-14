# Artificial Intelligence Engineer — Coursera

This repository contains code, notebooks, and resources for the **Artificial Intelligence Engineer** course on Coursera (IBM). It serves as a personal reference and portfolio of hands-on labs completed throughout the program.

## 📂 Repository Structure

```
artificial-intelligence-engineer-coursera/
├── README.md
├── .gitignore
└── advanced-deep-learning-with-pytorch/
    ├── 5 2 2bad_inshilization_logistic_regression_with_mean_square_error_v2.ipynb
    ├── 5 3_cross_entropy_logistic_regression_v2.ipynb
    └── 5 4softmax_in_one_dimension_v2.ipynb
```

## 📓 Notebooks

### Advanced Deep Learning with PyTorch

| Notebook | Topic | Description |
|----------|-------|-------------|
| [Bad Initialization — Logistic Regression with MSE](advanced-deep-learning-with-pytorch/5%202%202bad_inshilization_logistic_regression_with_mean_square_error_v2.ipynb) | Logistic Regression & Bad Initialization | Demonstrates how poor weight initialization (w=-5, b=-10) combined with Mean Square Error loss leads to suboptimal convergence — achieving only ~60% accuracy on a binary classification task. |
| [Cross Entropy — Logistic Regression](advanced-deep-learning-with-pytorch/5%203_cross_entropy_logistic_regression_v2.ipynb) | Logistic Regression & Cross-Entropy Loss | Shows how using Cross-Entropy loss instead of MSE enables convergence even with the same bad initialization values (w=-5, b=-10), achieving 100% accuracy. Highlights the importance of choosing the right loss function. |
| [Softmax Classifier in 1D](advanced-deep-learning-with-pytorch/5%204softmax_in_one_dimension_v2.ipynb) | Softmax Classifier | Builds a Softmax classifier using PyTorch's Sequential module to classify three linearly separable classes in one dimension. Demonstrates multi-class classification with CrossEntropyLoss and probability interpretation via Softmax. |

### Key Takeaways

- **Initialization matters**: Bad initial weight values can cause gradient descent to get stuck in suboptimal regions.
- **Loss function selection matters**: Cross-Entropy loss is better suited for classification problems than MSE, as it produces larger gradients for misclassified samples, helping the model escape poor initialization.
- **Softmax for multi-class classification**: The Softmax function converts model outputs into probabilities that sum to 1, making it ideal for multi-class problems.
- The first two notebooks use the same dataset and model architecture, making the comparison between MSE and Cross-Entropy loss direct and clear.

## 🛠️ Requirements

The notebooks use the following Python libraries:

- **Python** 3.12+
- **PyTorch** 2.8.0 (CPU)
- **NumPy**
- **Matplotlib** (including `mpl_toolkits.mplot3d` for 3D surface plots)

### Installation

```bash
pip install numpy matplotlib
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

Or install via the notebook's built-in `%pip install` commands.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/mhajjaj/artificial-intelligence-engineer-coursera.git
   cd artificial-intelligence-engineer-coursera
   ```

2. Open a notebook in Jupyter:
   ```bash
   jupyter notebook "advanced-deep-learning-with-pytorch/5 3_cross_entropy_logistic_regression_v2.ipynb"
   ```

3. Run the cells sequentially to reproduce the results.

## 📜 License

The lab content is © IBM Corporation. All rights reserved.  
This repository is for educational purposes as part of the Coursera AI Engineer program.

---

*Course: [IBM Artificial Intelligence Engineer](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera*