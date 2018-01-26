# VectorFlux
VectorFlux is a Neural Network library developed in Python. Here, we tried to create an easy to use library, that somewhat implements the same interface as Neural Network library Keras. Using the simplicity of Keras as reference, we implement a library VectorFlux. Creating a neural network with 800 + 800 HU's and a dropout layer can be as simple as shown below. Next we will elaborate on some of the (design) features that are implemented by the network.

```
vf = VectorFlux()
vf.add(Dense(800, activation='sigmoid', input_shape=784, optimizer='ADAM'))
vf.add(Dropout(0.5, input_shape=800))
vf.add(Dense(800, activation='sigmoid', input_shape=800, optimizer='ADAM'))
vf.add(Dense(10, activation='sigmoid', input_shape=800, optimizer='ADAM'))

vf.train(x_train, y_train, epochs=100000, alpha=0.001, mini_batch_size=130)
```

For more information about the library, we kindly refer to the accompanied report.
