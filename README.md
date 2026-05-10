# PROJECT BASED ON: 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities 
## PAPER LINK: https://arxiv.org/abs/2503.14858
## ORIGINAL REPO LINK: https://github.com/wang-kevin3290/scaling-crl


# Installation

## To start installation install `uv` and run:

```sh
uv sync
```

### If you're on Windows and you get:
```
Failed to build `pytinyrenderer==0.0.14`
```
You probably need Microsoft Visual C++ 14.0 or greater, because pytinyrenderer is written in C++.

1) Go to: https://visualstudio.microsoft.com/pl/visual-cpp-build-tools/
2) Download and install the build tools.
3) In build tools select: "Desktop development with C++". In optional (on the right) select also `Windows 10/11 SDK` and `MSVC v143 - VS 2022 C++ x64/x86`.

After installation `uv sync` should work. 
Then just fix the two Brax issues described below, and you'll be all set.


## Fixing two bugs in brax 0.10.1
1. There is a minor bug in brax's contact.py file. To fix it, first locate the brax contact.py file in your virtual environment: 

`Linux`:
```
find .venv -name contact.py
```

`Windows`:
```
Get-ChildItem -Path .venv -Filter contact.py -Recurse
```


Then open the file and replace it with the following code:
```python
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts.
    Args:
        sys: system defining the kinematic tree and other properties
        x: link transforms in world frame
    Returns:
        Contact pytree
    """
    #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
    # ncon = mjx.ncon(sys)
    # if not ncon:
    #   return None
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    # pytype: disable=wrong-arg-types
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    # pytype: enable=wrong-arg-types
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
```
2. There is also a minor bug in brax's json.py file. To fix it, first locate the brax json.py file in your virtual environment:

`Linux:`
```
find .venv -name json.py | grep "/brax/io/json.py"
```

`Windows:`
```
Get-ChildItem -Path .venv -Filter json.py -Recurse | Where-Object { $_.FullName -like "*\brax\io\json.py" }
```
Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```


# Running experiments
Now, we are ready to run the train script. To run the code, you'll need a GPU. (Possible on CPU, but really long). Reasonable parameters below: 

```sh
uv run train.py --env_id "ant" --eval_env_id "ant" --num_epochs 10 --total_env_steps 300000 --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --vis_length 1000  --save_buffer 0  --num_envs 16 --min_replay_size 2000 --unroll_length 20 
```
