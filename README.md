# C++ Web Server and Neural Network

## Data Download
1.  **Kaggle API Token:** Download from Kaggle.
2.  **Download Dataset:**
    ```bash
    kaggle datasets download hojjatk/mnist-dataset
    ```

## MNIST Image File Structure
* Header (16 bytes): Magic number, image count, rows, columns (big-endian).
* Pixel data follows the header.