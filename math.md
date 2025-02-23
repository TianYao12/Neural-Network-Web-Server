## Notation
L - Loss function (in this case cross-entropy).

a<sub>j</sub>(l) - Activation of neuron *j* in layer *l*.

W<sub>jk</sub>(l) - Weight connecting neuron *k* in layer *l-1* to neuron *j* in layer *l*.

b<sub>j</sub>(l) - Bias for neuron *j* in layer *l*.

z<sub>j</sub>(l) - Pre-activation value for neuron *j* in layer *l*. It is calculated as:

$$
z_j^{(l)} = (\sum_k W_{jk}^{(l)} a_k^{(l-1)}) + b_j^{(l)}
$$

f<sup>(l)</sup>(z) = Activation function in layer *l* (ReLU for hidden layers, Softmax for output layer).

δ<sub>j</sub>(l) = Error term for neuron *j* in layer *l*, defined as:

$$
\delta_j^{(l)} = \frac{\partial L}{\partial z_j^{(l)}}
$$

## Output Loss
Softmax activation is used for output along with cross-entropy loss:

$$
L = - \sum_j y_j \log a_j^{(L)}
$$

Taking the derivative w.r.t. the output activation:

$$
\frac{\partial L}{\partial a_j^{(L)}} = a_j^{(L)} - y_j
$$

$$
a_j^{(L)} = f^{(L)}\bigl(z_j^{(L)}\bigr)
$$


$$
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} = \frac{\partial}{\partial z_j^{(L)}} f^{(L)}\bigl(z_j^{(L)}\bigr) = f'^{(L)}\bigl(z_j^{(L)}\bigr)
$$

Using the chain rule:
$$
\delta_j^{(L)} = \frac{\partial L}{\partial z_j^{(L)}} = \frac{\partial L}{\partial a_j^{(L)}} \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} = (a_j^{(L)} - y_j) f'^{(L)}(z_j^{(L)})
$$

Since Softmax + Cross-Entropy simplifies to:

$$
\delta_j^{(L)} = a_j^{(L)} - y_j
$$

Code segment:
```c++
for (size_t j = 0; j < outputLayerProbability.size(); ++j)
{
    // (predicted probability for class j) - (one-hot-encoded actual label (correct 1, incorrect 0))
    output_error[j] = outputLayerProbability[j] - (j == actualLabel ? 1.0 : 0.0);
}

```

## Gradient Descent for Lth layer (Hidden-to-Output)
Recall that the pre-activation is defined as:

$$
z_j^{(L)} = \sum_k W_{jk}^{(L)} a_k^{(L-1)} + b_j^{(L)}
$$

$$
\frac{\partial z_j^{(L)}}{\partial W_{jk}^{(L)}} = a_k^{(L-1)}
$$

Combining both results, the weights between hidden and output layers are updated:

$$
\frac{\partial L}{\partial W_{jk}^{(L)}} = \frac{\partial L}{\partial z_j^{(L)}}\frac{\partial z_j^{(L)}}{\partial W_{jk}^{(L)}} = \delta_j^{(L)} \cdot a_k^{(L-1)}
$$

Using gradient descent to update the weights:

$$
W_{jk}^{(L)} \gets W_{jk}^{(L)} - \eta \cdot \delta_j^{(L)} \cdot a_k^{(L-1)}
$$

Using gradient descent to update the biases:

$$
b_j^{(L)} \gets b_j^{(L)} - \eta \cdot \delta_j^{(L)}
$$

Code segment:
```c++
for (size_t j = 0; j < hiddenToOutputLayerWeights.size(); ++j)
{
    for (size_t k = 0; k < hiddenToOutputLayerActivation.size(); ++k)
    {
        hiddenToOutputLayerWeights[j][k] -= learningRate * output_error[j] * hiddenToOutputLayerActivation[k];
    }
    outputLayerBiases[j] -= learningRate * output_error[j];
}
```

## Backpropagation
To propagate the error backward, we use the **chain rule** to compute:

$$
\delta_j^{(L-1)} = \frac{\partial L}{\partial z_j^{(L-1)}} = \sum_k \frac{\partial L}{\partial z_k^{(L)}} \cdot \frac{\partial z_k^{(L)}}{\partial z_j^{(L-1)}}
$$

$$
z_k^{(L)} = \sum_j W_{kj}^{(L)} a_j^{(L-1)} + b_k^{(L)}
$$

$$
\frac{\partial z_k^{(L)}}{\partial z_j^{(L-1)}} = W_{kj}^{(L)} f'^{(L-1)}(z_j^{(L-1)})
$$

Substituting these results into the chain rule equation:

$$
\delta_j^{(L-1)} = \sum_k \delta_k^{(L)} W_{kj}^{(L)} \cdot f'^{(L-1)}(z_j^{(L-1)})
$$

Thus, the error for neuron *j* in layer *L-1* is:

$$
\delta_j^{(L-1)} = f'^{(L-1)}(z_j^{(L-1)}) \sum_k W_{kj}^{(L)} \delta_k^{(L)}
$$

Since the hidden layer uses ReLU, we need to adjust for its derivative:

$$
\delta_j^{(L-1)} \gets \delta_j^{(L-1)} \cdot f'^{(L-1)}(z_j^{(L-1)})
$$

For ReLU:

$$
f'(z) = \begin{cases} 
1, & z > 0 \\ 
0, & z \leq 0 
\end{cases}
$$

Code segment:
```c++
vector<double> hidden_error(hiddenToOutputLayerActivation.size(), 0.0);
for (size_t j = 0; j < hiddenToOutputLayerActivation.size(); ++j)
{
    hidden_error[j] = 0.0;

    for (size_t k = 0; k < outputLayerProbability.size(); ++k)
    {
        hidden_error[j] += output_error[k] * hiddenToOutputLayerWeights[k][j];
    }
    hidden_error[j] *= NNUtils::ActivationFunctions::reluDerivative(hiddenToOutputLayerActivation[j]);
}
```

## Gradient Descent for (L-1)th layer (Input-to-Hidden)
Using the same gradient descent rule:

$$
\frac{\partial L}{\partial W_{jk}^{(L-1)}} = \delta_j^{(L-1)} \cdot a_k^{(L-2)}
$$

Weight update:

$$
W_{jk}^{(L-1)} \gets W_{jk}^{(L-1)} - \eta \cdot \delta_j^{(L-1)} \cdot a_k^{(L-2)}
$$

Bias update:

$$
b_j^{(L-1)} \gets b_j^{(L-1)} - \eta \cdot \delta_j^{(L-1)}
$$

Code segment:
```c++
for (size_t j = 0; j < inputToHiddenLayerWeights.size(); ++j)
{
    for (size_t k = 0; k < inputNormalized.size(); ++k)
    {
        // since we have 1 hidden layer, the activation in (L-2) layer is the input normalized
        inputToHiddenLayerWeights[j][k] -= learningRate * hidden_error[j] * inputNormalized[k];
    }
    hiddenLayerBiases[j] -= learningRate * hidden_error[j];
}
```