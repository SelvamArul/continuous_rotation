# To install

```
python setup.py install
```

# create an identity rotation


```python
import torch

rotation = torch.eye(3)
print (rotation)
```

    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])


# create 6D representation of the rotation matrix by dropping the last column vector


```python
rotation_6d =  rotation.flatten()[:6].view(-1, 6) # (bacth x 6)
print (rotation_6d)
```

    tensor([[1., 0., 0., 0., 1., 0.]])

# recreate rotation matrix from 6D vector


```python
import continuous_rotation as cr
cr.compute_rotation_matrix_from_ortho6d(rotation_6d)
```




    tensor([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])

# create 5D represtation from 6D (uses stereographic projection internally)

```python
rotation_5d = cr.compute_5D_from_6D(rotation_6d) # (batch x 6) -> (batch x 5)
print (rotation_5d)
```
    tensor([[ 7.0711e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  7.0711e-01,
         -2.9802e-08]])

# convert 5D back to 6D 

```python
rotation_6d_ = cr.compute_6D_from_5D(rotation_5d)
print (rotation_6d_.round()) # round() only for readability
```
    tensor([[1., 0., 0., 0., 1., -0.]])


