# C++ Web Server and Neural Networks
## Purpose
- I wanted to get a more fundamental understanding of neural networks and web servers, which we use all the time using abstracted frameworks like Node.js or PyTorch
- This web server take requests and respond with neural network data (weights, biases, loss, prediction) after each epoch
- A frontend will make requests to the server, and display the data, visualizing how the various parameters of a neural network change over time.

## Versions
- On the main branch is the implementation of the standard feed forward neural network
- I tried with a convolutional neural network, and that needs some work. It is on the cnn branch

## Saving Training and Inference Data
- After each epoch, the **weights** and **biases** are saved in binary
- After inference, the output probabilities are also saved

## Running
* Build & Run Training/Inference:
    ```bash
    (cd backend/networking/NN && make clean && make && ./train.out && ./inference.out)
    ```
* Build & Run Server:
    ```bash
    (cd backend/networking && make clean && make && ./server.exe)
    ```
* Run Frontend:
    ```bash
    (cd frontend && npm run dev)
    ```

## Socket and Servers
A socket is a software endpoint that enables communication between two computers over a network.

### Socket classes
- SimpleSocket: Creates a raw socket. 
- BindingSocket
    - Inherits from SimpleSocket
    - Binds the socket to an IP and port. 
    - A raw socket does not do anything on its own. Binding allows clients to know where to connect. 
- ListeningSocket
    - Inherits from BindingSocket
    - Bound sockets cannot accept connections until it's actively listening
    - listen() sets up a queue for incoming connections
- ConnectingSocket: A client-side socket that connects to a server. This is not used right now as I'm just connecting to the server from my Next.js frontend

### Server Classes
- SimpleServer: Owns the listening socket and handles client requests.
- TestServer: A specific implementation of SimpleServer that processes requests.

### Web Server Flow
- Create a socket 
- Bind it to an address and port
- Listen for incoming connections 
- Accept client connections and respond

## MNIST Image File Structure
* Header (16 bytes): Magic number, image count, rows, columns (big-endian).
* Pixel data of the actual images follows the header.

## Data Download
1.  **Kaggle API Token:** Download from Kaggle.
2.  **Download Dataset:**
    ```bash
    kaggle datasets download hojjatk/mnist-dataset
    ```