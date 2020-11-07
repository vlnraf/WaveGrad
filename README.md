# WaveGrad 

WaveGrad is a tiny Python library to create and train deep feedforward neural netwroks.

## Getting Start

It's possible to create different networks with different topology. You just need to import the [`Sequential`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.network) module and add [`Layers`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.layers) into it:

```python
from neuralnet.network import Sequential
from neuralnet.layers import LayerDense
from neuralnet.activations import *
from neuralnet.losses import *
from neuralnet.optimizers import *

# network
net = Sequential()
net.add(LayerDense(17, 5, sigmoid))
# You can always add more layers
net.add(LayerDense(5, 1, tanh))
# You can alway add optimizer and loss function
optim = GD(net.layers, lr=0.01, momentum=0.9)
net.use(mse)
# train
net.fit(Xtrain, ytrain, epochs=500, optimizer=optim)

```

See [`activations`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.activations) for the list of activation functions.
See [`optimizers`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.optimizers) for the list of optimizers.
See [`losses`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.losses) for the list of losees functions.
