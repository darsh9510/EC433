# TIM for Few-Shot Learning

This project is an implementation of the NeurIPS 2020 paper, "Transductive Information Maximization For Few-Shot Learning" [1].

## Project Structure

-   `model.py`: Defines the ResNet-18 feature extractor and classifier head.
-   `utils.py`: Contains helper functions, including the core TIM loss components (marginal and conditional entropy) and label smoothing.
-   `train_base.py`: Script for **Component 1**. Trains the base feature extractor on the (stubbed) base dataset using standard cross-entropy with label smoothing.
-   `tim_inference.py`: Script for **Component 2**. Implements the transductive inference (TIM-GD) on (stubbed) few-shot tasks.
-   `requirements.txt`: Required Python packages.

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the base training to create a checkpoint:
    ```bash
    python train_base.py
    ```
    This will create a file named `base_model.pth`.

3.  Run the TIM-GD inference using the trained model:
    ```bash
    python tim_inference.py
    ```