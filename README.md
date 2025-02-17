# C++ Web Server and Neural Network

## Purpose
- The idea is to implement a web server in C++ from scratch that can take requests and respond with neural networok data (weights, biases, loss, prediction) after each epoch
- A Next.js frontend will make requests to the server, and display the data, visualizing how the various parameters of a neural network change over time.

## Data Download
1.  **Kaggle API Token:** Download from Kaggle.
2.  **Download Dataset:**
    ```bash
    kaggle datasets download hojjatk/mnist-dataset
    ```

## MNIST Image File Structure
* Header (16 bytes): Magic number, image count, rows, columns (big-endian).
* Pixel data of the actual images follows the header.

## Saved Data
- After each epoch, the **weights** and **biases** are saved in binary
- After inference, the output probabilities are also saved

## Runnning
* **Build & Run Training/Inference:**
    ```bash
    (cd hdelibc/networking/NN && make clean && make && ./train.out && ./inference.out)
    ```
* **Build & Run Server:**
    ```bash
    (cd hdelibc/networking && make clean && make && ./server.exe)
    ```
* **Run Frontend:**
    ```bash
    (cd frontend && npm run dev)
    ```