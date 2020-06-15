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


