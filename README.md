# Cabinet Classification Project

This project implements a cabinet type classifier for architectural drawings using a ResNet50 model.

## Project Structure

- `train.py`: Script to train the model.
- `inference.py`: Script to run inference on images using the trained model.
- `evaluate_test_set.py`: Script to reproduce the test set evaluation results (metrics & confusion matrix).
- `best_model.pth`: The saved model weights (ResNet50).
- `Dockerfile`: Configuration to containerize the application.
- `requirements.txt`: Python dependencies.

## Setup & Installation

### Option 1: Using Docker (Recommended)

1.  **Build the Docker image:**
    ```bash
    docker build -t cabinet-classifier .
    ```

2.  **Run Inference:**
    To run inference on a local image or folder, you need to mount the volume.
    
    *   **Single Image:**
        Assume you have an image `test.jpg` in your current folder.
        ```bash
        docker run -v $(pwd):/app/data cabinet-classifier python inference.py /app/data/test.jpg
        ```
    
    *   **Folder:**
        ```bash
        docker run -v $(pwd)/dataset:/app/dataset cabinet-classifier python inference.py /app/dataset/lc_bcabo
        ```

3.  **Run Training:**
    To train a new model (requires the `dataset` folder to be present/mounted):
    ```bash
    docker run -v $(pwd)/dataset:/app/dataset cabinet-classifier python train.py --data_dir /app/dataset --epochs 5
    ```

3.  **Run Evaluation (Reproduce Results):**
    To see the metrics on the test set (same as the notebook):
    ```bash
    docker run -v $(pwd)/dataset:/app/dataset cabinet-classifier python evaluate_test_set.py --data_dir /app/dataset
    ```

### Option 2: Local Python Environment

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Inference:**
    ```bash
    python inference.py path/to/image.jpg
    ```

3.  **Run Training:**
    ```bash
    python train.py --data_dir dataset/
    ```

4.  **Run Evaluation:**
    ```bash
    python evaluate_test_set.py
    ```

## Approach

The model uses a **ResNet50** architecture, fine-tuned for the 5 cabinet classes.
- **Preprocessing**: Images are padded to be square (preserving aspect ratio) before resizing to 224x224. This prevents distortion of the architectural lines.
- **Training**: Uses stratisfied splitting to handle class imbalance, and a weighted CrossEntropyLoss.

## Classes
- `lc:bcabo`: Base Cabinet - Open
- `lc:wcabo`: Wall Cabinet - Open
- `lc:muscabinso`: Miscellaneous Cabinet - Insulated
- `lc:wcabcub`: Wall Cabinet - Open Cubbie
- `lc:bcabocub`: Base Cabinet - Open Cubbie
