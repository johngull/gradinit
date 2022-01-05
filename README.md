# PyTorch implementation of the GradInit

Pytorch-based library to initialize any neural network with the gradient descent and your dataset.
This implementation is the simple way to use the method described in the paper [Chen Zhu et al. GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training](https://arxiv.org/abs/2102.08098)

Gradinit may be used as the alternative to the warmup mechanism.

## Installation

    pip install gradinit

or

    pip install --upgrade git+https://github.com/johngull/gradinit.git

## Differences from the original paper

This implementation uses simplified loss in the case of the small gradients.
in this case, `gradinit` uses the user's loss instead of the complex 2-stage procedure described in the paper.

## Usage

To make gradinit you need to wrap your model with the `GradInitWrapper` 
and then you can use the usual training loop with some differences:
- recalculate loss with the `GradInitWrapper.grad_loss` function
- clip scales with the `GradInitWrapper.clamp_scales` function

Here is a simplified example of the full gradinit process:
```python
model: torch.nn.Module = ...

with GradInitWrapper(model) as ginit:
    #it is important to create optimizer after wraping your model
    optimizer = torch.optim.Adam(model.parameters())  
    norm = 1    # use 2 for the SGD optimizer

    model.train()
    for x, y in data_loader:
        pred = model(x)
        loss = criterion(pred, y)
    
        # recalculate gradinit loss based on your loss
        loss = ginit.grad_loss(loss, norm)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # clip minimum values for the init scales
        ginit.clamp_scales()
    
# on exit of with statement model is recovered to its normal way
# here you should create your main optimizer and start training
optimizer = torch.optim.Adam(model.parameters())
...

```

Alternatively to the `with`-notation 
you may use `detach` or delete the wrapper object after finishing the initialization process.
```python
ginit = GradInitWrapper(model)
# full gradinit loop
...
ginit.detach() # or del ginit

# here you should create your main optimizer and start training
optimizer = torch.optim.Adam(model.parameters())
...

```
