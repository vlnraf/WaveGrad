# WaveGrad 

<p align="center">
    <img width="300" src = "https://github.com/vlnraf/WaveGrad/blob/master/docs/static/logo.webp">
</p>

---
WaveGrad is a tiny Python library to create and train deep feedforward neural netwroks. [Documentation](https://vlnraf.github.io/WaveGrad/build/html/index.html)

## Getting Start

It's possible to create different networks with different topology. You just need to import the [`Sequential`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.network) module and add [`Layers`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.layers) into it:

```python
from wavegrad.network import Sequential
from wavegrad.layers import LayerDense
from wavegrad.activations import *
from wavegrad.losses import MSE
from wavegrad.optimizers import *

# network
net = Sequential()
net.add(LayerDense(17, 5, sigmoid))
# You can always add more layers
net.add(LayerDense(5, 1, tanh))
# You can alway add optimizer and loss function
optim = GD(net.layers, lr=0.01, momentum=0.9)
net.use(MSE)
# train
net.fit(Xtrain, ytrain, epochs=500, optimizer=optim)

```

See [`activations`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.activations) for the list of activation functions.
See [`optimizers`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.optimizers) for the list of optimizers.
See [`losses`](https://vlnraf.github.io/WaveGrad/build/html/api.html#module-wavegrad.losses) for the list of losees functions.

## TODO

- implement the validation split from yourself without use the library sklearn
