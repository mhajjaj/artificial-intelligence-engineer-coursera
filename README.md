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
    ├── 5 4softmax_in_one_dimension_v2.ipynb
    ├── 6 2lab_predicting _MNIST_using_Softmax_v2.ipynb
    ├── 7 1_simple1hiddenlayer.ipynb
    ├── 7 2multiple_neurons.ipynb
    ├── 7 3xor_v2.ipynb
    ├── 7 4one_layer_neural_network_MNIST.ipynb
    ├── 7 5 1activationfuction_v2.ipynb
    ├── 7 5 2mist1layer_v2.ipynb
    ├── 8 1 1mist2layer_v2.ipynb
    ├── 8 1 2mulitclassspiralrulu_v2.ipynb
    ├── 8 2 1dropoutPredictin_v2.ipynb
    ├── 8 2 2dropoutRegression_v2.ipynb
    ├── 8 3 1 initializationsame.ipynb
    ├── 8 3 2Xaviermist1layer_v2.ipynb
    ├── 8 3 3 He_Initialization_v2.ipynb
    ├── 9 1What_is_Convolution.ipynb
    ├── 9 2Activation_max_pooling 1.ipynb
    ├── 9 3Multiple Channel Convolution.ipynb
    ├── 9 4 1ConvolutionalNeralNetworkSimple example.ipynb
    ├── 9 4 2CNN_Small_Image.ipynb
    ├── 9 4 3CNN_Small_Image_batch.ipynb
    └── data/
```

## 📓 Notebooks

### Advanced Deep Learning with PyTorch

| Notebook | Topic | Description |
|----------|-------|-------------|
| [Bad Initialization — Logistic Regression with MSE](advanced-deep-learning-with-pytorch/5%202%202bad_inshilization_logistic_regression_with_mean_square_error_v2.ipynb) | Logistic Regression & Bad Initialization | Demonstrates how poor weight initialization (w=-5, b=-10) combined with Mean Square Error loss leads to suboptimal convergence — achieving only ~60% accuracy on a binary classification task. |
| [Cross Entropy — Logistic Regression](advanced-deep-learning-with-pytorch/5%203_cross_entropy_logistic_regression_v2.ipynb) | Logistic Regression & Cross-Entropy Loss | Shows how using Cross-Entropy loss instead of MSE enables convergence even with the same bad initialization values (w=-5, b=-10), achieving 100% accuracy. Highlights the importance of choosing the right loss function. |
| [Softmax Classifier in 1D](advanced-deep-learning-with-pytorch/5%204softmax_in_one_dimension_v2.ipynb) | Softmax Classifier | Builds a Softmax classifier using PyTorch's Sequential module to classify three linearly separable classes in one dimension. Demonstrates multi-class classification with CrossEntropyLoss and probability interpretation via Softmax. |
| [Predicting MNIST using Softmax](advanced-deep-learning-with-pytorch/6%202lab_predicting%20_MNIST_using_Softmax_v2.ipynb) | Softmax on MNIST | Applies a single-layer Softmax classifier to the MNIST handwritten digit dataset. Covers data loading with `torchvision`, model training with CrossEntropyLoss, validation accuracy tracking, parameter visualization, and analysis of misclassified vs. correctly classified samples. |
| [Simple One Hidden Layer](advanced-deep-learning-with-pytorch/7%201_simple1hiddenlayer.ipynb) | Neural Networks | Introduces neural networks with a single hidden layer (2 neurons) to classify non-linearly separable 1D data. Demonstrates how the hidden layer transforms data into a linearly separable space, enabling classification that a simple logistic regression cannot achieve. |
| [Neural Networks More Hidden Neurons](advanced-deep-learning-with-pytorch/7%202multiple_neurons.ipynb) | Neural Networks | Explores neural networks with multiple hidden neurons (9 neurons) to classify complex non-linearly separable data with two separate regions. Uses BCELoss and Adam optimizer to learn XOR-like patterns in 1D data. |
| [Noisy XOR](advanced-deep-learning-with-pytorch/7%203xor_v2.ipynb) | Neural Networks | Investigates how many neurons are needed to classify noisy XOR data with one hidden layer. Experiments with 1, 2, and 3 neurons, demonstrating that XOR requires at least 2 neurons in the hidden layer to be solved effectively. |
| [One Hidden Layer Neural Network on MNIST](advanced-deep-learning-with-pytorch/7%204one_layer_neural_network_MNIST.ipynb) | Neural Networks on MNIST | Applies a neural network with one hidden layer (100 neurons) to classify MNIST handwritten digits. Compares the custom module approach with `nn.Sequential`, showing improved accuracy over single-layer softmax classifiers. |
| [Activation Functions](advanced-deep-learning-with-pytorch/7%205%201activationfuction_v2.ipynb) | Activation Functions | Compares different activation functions (Sigmoid, Tanh, ReLU) in PyTorch. Demonstrates how to use both module-based (`nn.Sigmoid`, `nn.Tanh`, `nn.ReLU`) and function-based (`torch.sigmoid`, `torch.tanh`, `torch.relu`) approaches. |
| [Testing Activation Functions on MNIST](advanced-deep-learning-with-pytorch/7%205%202mist1layer_v2.ipynb) | Activation Functions on MNIST | Compares the performance of Sigmoid, Tanh, and ReLU activation functions on the MNIST dataset using a neural network with one hidden layer (100 neurons). Demonstrates that ReLU typically converges faster and achieves better accuracy. |
| [Deep Neural Networks — Two Hidden Layer](advanced-deep-learning-with-pytorch/8%201%201mist2layer_v2.ipynb) | Neural Networks | Trains a neural network with two hidden layers to classify the MNIST dataset. Compares Sigmoid, Tanh, and ReLU activations, demonstrating how deeper architectures improve representation power. |
| [Deeper Neural Networks with ModuleList](advanced-deep-learning-with-pytorch/8%201%202mulitclassspiralrulu_v2.ipynb) | Deeper Networks | Creates a deeper neural network using `nn.ModuleList()` to dynamically build layers with variable hidden-unit counts. Trains on a multi-class spiral dataset with ReLU activation, covering training, validation, and result analysis. |
| [Dropout for Classification](advanced-deep-learning-with-pytorch/8%202%201dropoutPredictin_v2.ipynb) | Dropout | Applies dropout regularization to a classification model built the PyTorch way. Covers model creation, cost function definition, and batch gradient descent to prevent overfitting. |
| [Dropout in Regression](advanced-deep-learning-with-pytorch/8%202%202dropoutRegression_v2.ipynb) | Dropout | Applies dropout regularization to a regression model. Demonstrates how dropout helps generalization by randomly dropping neurons during training in a regression context. |
| [Same Weights Initialization](advanced-deep-learning-with-pytorch/8%203%201%20initializationsame.ipynb) | Initialization | Shows the effect of initializing all weights to the same value (e.g., zeros or identical values). Demonstrates why symmetric initialization prevents learning and causes neurons to compute identical features. |
| [Xavier Initialization on MNIST](advanced-deep-learning-with-pytorch/8%203%202Xaviermist1layer_v2.ipynb) | Initialization | Compares Uniform, Default, and Xavier Uniform initialization strategies on MNIST with Tanh activation. Analyzes training accuracy and convergence speed of each method. |
| [He Initialization on MNIST](advanced-deep-learning-with-pytorch/8%203%203%20He_Initialization_v2.ipynb) | Initialization | Compares Uniform, Default, and He initialization strategies on MNIST with ReLU activation. Demonstrates that He initialization is designed for ReLU and improves training stability. |
| [What is Convolution?](advanced-deep-learning-with-pytorch/9%201What_is_Convolution.ipynb) | Convolution | Introduces 2D convolution with `nn.Conv2d`: kernel size, stride, padding, and output shape formulas. Includes a hands-on exercise computing output values without running the function. |
| [Activation and Max Pooling](advanced-deep-learning-with-pytorch/9%202Activation_max_pooling%201.ipynb) | Pooling | Explores activation functions followed by max pooling in CNNs. Covers how pooling reduces spatial dimensions while preserving dominant features, and its impact on model performance. |
| [Multiple Input and Output Channels](advanced-deep-learning-with-pytorch/9%203Multiple%20Channel%20Convolution.ipynb) | Multi-Channel CNNs | Extends convolution to multiple input and output channels, explaining how filters produce feature maps and how multiple filters build depth in CNNs. |
| [CNN Simple Example](advanced-deep-learning-with-pytorch/9%204%201ConvolutionalNeralNetworkSimple%20example.ipynb) | CNN Basics | Builds a simple CNN to classify horizontal vs. vertical lines. Covers data preparation, custom model class, training loop, and evaluation for a minimal image classification task. |
| [CNN with Small Images](advanced-deep-learning-with-pytorch/9%204%202CNN_Small_Image.ipynb) | Image CNN | Constructs a CNN for small image classification. Walks through dataset preparation, model architecture with convolution + pooling layers, training, and validation. |
| [CNN with Batch Normalization](advanced-deep-learning-with-pytorch/9%204%203CNN_Small_Image_batch.ipynb) | Batch Normalization | Compares a standard CNN against one using Batch Normalization. Demonstrates how batch norm stabilizes training, allows higher learning rates, and improves model accuracy. |

### Key Takeaways

- **Initialization matters**: Bad initial weight values can cause gradient descent to get stuck in suboptimal regions.
- **Loss function selection matters**: Cross-Entropy loss is better suited for classification problems than MSE, as it produces larger gradients for misclassified samples, helping the model escape poor initialization.
- **Softmax for multi-class classification**: The Softmax function converts model outputs into probabilities that sum to 1, making it ideal for multi-class problems.
- **Scaling to real data**: The MNIST notebook demonstrates how the same Softmax + CrossEntropyLoss approach scales from synthetic 1D data to real-world image classification (28x28 grayscale digits, 10 classes).
- **Hidden layers enable non-linear classification**: Neural networks with hidden layers can classify non-linearly separable data (like XOR) that single-layer models cannot.
- **Neuron count matters**: The XOR notebook shows that complex patterns require sufficient neurons in the hidden layer — 1 neuron fails, but 2+ neurons succeed.
- **Activation function choice impacts performance**: ReLU typically converges faster and achieves better accuracy than Sigmoid or Tanh, especially for deeper networks and larger datasets like MNIST.
- **Deeper networks**: Adding more hidden layers increases representation power. `nn.ModuleList()` allows dynamic layer construction.
- **Dropout prevents overfitting**: Randomly dropping neurons during training forces the network to learn redundant representations, improving generalization on both classification and regression tasks.
- **Initialization matters for deep networks**: Same/zero weight initialization causes symmetry and prevents learning. Xavier initialization suits Tanh, while He initialization is optimal for ReLU.
- **Convolution extracts spatial features**: 2D convolution with `nn.Conv2d` uses kernels to detect patterns. Stride and padding control output size; $M_{new} = (M + 2 \times padding - K) / stride + 1$.
- **Pooling reduces dimensionality**: Max pooling preserves dominant features while reducing spatial dimensions, making the model more efficient.
- **Multi-channel CNNs**: Multiple input/output channels allow networks to learn diverse feature maps. Batch Normalization stabilizes training and improves convergence.
- The first two notebooks use the same dataset and model architecture, making the comparison between MSE and Cross-Entropy loss direct and clear.

## 🛠️ Requirements

The notebooks use the following Python libraries:

- **Python** 3.12+
- **PyTorch** 2.8.0 (CPU)
- **torchvision** 0.23.0 (for MNIST dataset loading and image transforms)
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