# WaveGrad 

WaveGrad is a tiny Python library to create and train deep feedforward neural netwroks.

## Getting Start

It's possible to create different networks with different topology. You just need to import the [`Sequential`](https://github.com/vlnraf/WaveGrad/blob/master/neuralnet/network.py) module and add [`Layers`](https://github.com/vlnraf/WaveGrad/blob/master/neuralnet/layers.py) into it:

```python
from neuralnet.network import Sequential
from neuralnet.layers import LayerDense

# network
net = Sequential()
net.add(LayerDense(17, 5, sigmoid))
# You can always add more layers
net.add(LayerDense(5, 1, tanh))
```

See [`activations`](https://github.com/vlnraf/WaveGrad/blob/master/neuralnet/activations.py) for the list of activation functions.
