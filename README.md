# Molecule Toxicity Classifier

This project aims to classify molecules based on their toxicity using a deep learning model, specifically a 1D Convolutional Neural Network (CNN). The dataset includes both toxic and non-toxic molecules, and the goal is to accurately predict the toxicity of molecules based on their SMILES representation.

## Dataset

The project uses two CSV files containing molecular data:

- **`train_names_smiles.csv`**: Training data with molecule names and their corresponding SMILES strings.
- **`train_names_labels.csv`**: Training data with molecule names and their toxicity labels (1 for toxic, 0 for non-toxic).
- **`test_names_smiles.csv`**: Testing data with molecule names and their corresponding SMILES strings.
- **`test_names_labels.csv`**: Testing data with molecule names and their toxicity labels.

### Sample Data

#### `train_names_smiles.csv`
| name       | smiles         |
|------------|----------------|
| Molecule1  | C1=CC=CC=C1    |
| Molecule2  | CC(C)C(=O)O    |

#### `train_names_labels.csv`
| name       | toxicity |
|------------|----------|
| Molecule1  | 1        |
| Molecule2  | 0        |

## Model

The model is built using a 1D Convolutional Neural Network (CNN) with the following architecture:

1. **Conv1 Layer**: 1D convolution layer with 32 filters.
2. **Conv2 Layer**: 1D convolution layer with 64 filters.
3. **Conv3 Layer**: 1D convolution layer with 128 filters.
4. **Fully Connected Layers**: After convolution, the output is flattened and passed through fully connected layers.

The model is trained to minimize binary cross-entropy loss using the Adam optimizer.

### Key Features:

- **Dropout Layers**: To prevent overfitting, dropout layers are applied between convolutional layers.
- **Batch Normalization**: Batch normalization is applied after each convolution layer to stabilize and speed up training.
- **ReLU Activation**: ReLU is used as the activation function for all layers except the output layer.
- **Final Output**: The final layer uses a sigmoid function to output a probability between 0 and 1 (toxic or non-toxic).

## Training

To train the model, we use the following function:

```python
def train_cnn(X_train, y_train, input_length, num_epochs=20, batch_size=32, learning_rate=0.001, dropout_rate=0.3):
    # Function implementation
```

## Evaluation
After training the model, you can evaluate its performance using the test data. The model's accuracy and ROC-AUC score are computed. Additionally, the confusion matrix is displayed to show how well the model distinguishes between toxic and non-toxic molecules.

```python
accuracy = accuracy_score(y_test, binary_predictions)
roc_auc = roc_auc_score(y_test, predictions)
f1 = f1_score(y_test, binary_predictions)
```

## Confusion Matrix:
|                | Predicted Non-Toxic | Predicted Toxic |
|----------------|---------------------|-----------------|
| Actual Non-Toxic |         216         |        22       |
| Actual Toxic     |         20          |         7       |

## Conclusion

The current model, based on a 1D Convolutional Neural Network (CNN), shows some promise in classifying molecules as toxic or non-toxic. However, the performance metrics, including the F1-score, indicate that there is room for improvement. Here are some key conclusions and insights:

Key Insights for Improvement:
1. Class Imbalance:

- The model faces a class imbalance between toxic and non-toxic molecules, with significantly more non-toxic samples. Despite the augmentation of toxic samples, the F1-score is still low, suggesting that the model is not learning to identify toxic molecules well.
- Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic toxic samples, or try other sampling methods such as undersampling non-toxic molecules. Alternatively, experiment with class weights in the loss function to penalize misclassifying toxic molecules more heavily.

2. Model Architecture:

- The current model uses a relatively simple CNN with three convolution layers and fully connected layers. While this can work well for certain tasks, more complex architectures could improve performance.
- Experiment with deeper architectures, such as adding more convolutional layers or utilizing ResNet or DenseNet architectures. These models help in training deeper networks by addressing the vanishing gradient problem and improving feature extraction.

## Final Recommendations:
To improve the model’s performance in predicting toxic molecules:

- Address class imbalance more effectively.
- Experiment with more advanced models and architectures.


