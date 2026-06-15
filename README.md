# PROJECT BASED ON: 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities 
## PAPER LINK: https://arxiv.org/abs/2503.14858
## ORIGINAL REPO LINK: https://github.com/wang-kevin3290/scaling-crl

# GPU

To train models using this repo you will need GPU.

### Easy, but limited:
If you don't own one, a good option is to create https://lightning.ai/ account. You will get 15 credits every month to work with GPU, which is approximately 30 hours of work on GPU similar to collab.
For example using A100 ~ 6 hours.
`Try using lightning.ai only when training/local checks.`

### Bigger trainings:

`TBD`


# Installation

## To start installation install `uv` and run:

```sh
uv sync
```

### If you're on Windows and you get:
```
Failed to build `pytinyrenderer==0.0.14`
```
You probably need Microsoft Visual C++ 14.0 or greater, because `pytinyrenderer` is written in C++.

1) Go to: https://visualstudio.microsoft.com/pl/visual-cpp-build-tools/
2) Download and install the build tools.
3) In build tools select: `Desktop development with C++`. In optional (on the right) select also `Windows 10/11 SDK` and `MSVC v143 - VS 2022 C++ x64/x86`.

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

`By-hand`:
```
.venv/lib/brax/contact.py
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

`By-hand`:
```
.venv/lib/brax/io/json.py
```

Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```

# Logging

For loging and visulizations we use [wandb.ai](https://wandb.ai/). 

1) Create wandb account.
2) Create a project.
3) Create an API key.
4) Run: `uv run wandb login`.
5) Run: `wandb sync --project PROJECT-NAME --entity USERNAME PATH-TO-WANDB-LOGS `.

# Running experiments
Now, we are ready to run the train script. To run the code, you'll need a GPU. (Possible on CPU, but really long). Reasonable parameters below: 

1) Test if working.
```sh
uv run main.py --env_id "ant" --eval_env_id "ant" --num_epochs 10 --total_env_steps 300000 --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --vis_length 1000  --save_buffer 0  --num_envs 16 --min_replay_size 2000 --unroll_length 20 
```

2) More real scenario.
```sh
uv run main.py --env_id "humanoid" --eval_env_id "humanoid" --num_epochs 50 --total_env_steps 50000000 --critic_depth 32 --actor_depth 32 --actor_skip_connections 4 --critic_skip_connections 4 --vis_length 1000  --save_buffer 0  --num_envs 512 --min_replay_size 1000 --unroll_length 62
```

# Running on entropy

Create `logs` directory in the same directory as your `main.py`. Create a `job.sh` file (content below - example for a small run), and finally, run `sbatch job.sh`.

There might be a problem with vritual environment setup - more detailed instructions will be added soon. Short version: You have to create venv, run pip install uv, then uv sync, then add uv once again (I do not remember exact steps :P)

```sh
#!/bin/bash
#
#SBATCH --job-name=transformer-test
#SBATCH --partition= YOUR PARTITION
#SBATCH --qos= YOUR QOS
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-transformer-test.txt
#SBATCH --error=logs/slurm-transformer-test-error.txt
#SBATCH --time=30

echo "Started"

set -eux

uv run main.py \
   --env_id "ant" \
   --eval_env_id "ant" \
   --num_epochs 10 \
   --total_env_steps 300_000 \
   --critic_depth 16 \
   --actor_depth 16 \
   --actor_skip_connections 4 \
   --critic_skip_connections 4 \
   --vis_length 1000  \
   --save_buffer 0  \
   --num_envs 16 \
   --min_replay_size 2000 \
   --unroll_length 20

 echo "Finished"
```


# Transformers for NLP
To run transformers look at job-transformers.sh, it contains most of the important options that are available. Most important options:

- Pooling (transformer_pooling):
    1. cls - a learned cls token prepended to the sequence, probably the best approach
    2. mean - taking the mean of the tokens
    3. flatten - flattening the tokens and downsampling with a dense layer.
-  Tokenization (tokenization):
    1. patches - naive split into patches
    2. semantic - smart one, where each joint is a token + 1 token for body
    3. per_dim - every observation and action dimension is one token
- Transformer mode (transformer_mode):
    1. None - MLP
    2. State - only SA encoder is transformer
    3. StateGoal - whole critic is transformer
    4. StateActor - SA encoder and actor are transformer
    5. Full - all networks are transformers


## Code layout
Most of the basic transformer code is in modules/ directory, with utils having the base transformer block class.


# Goal encoders

The goal encoder produces a 64-dim representation of the target goal that is
compared (via InfoNCE loss) against the state–action encoder's output.
Several mutually exclusive options control which encoder is used:

### Frozen pretrained text models (``--text-encoder``)

When enabled (default), the goal is converted to a text prompt
``\"Your goal is (x,y)\"`` and encoded by a frozen HuggingFace BERT-family model.

The model is selected via ``--text-model``:

| Short name | HuggingFace model | Dim |
|------------|-------------------|-----|
| ``minilm`` | sentence-transformers/all-MiniLM-L6-v2 | 384 |
| ``bge``    | BAAI/bge-small-en-v1.5                 | 384 |
| ``gte``    | thenlper/gte-small                     | 384 |
| ``e5``     | intfloat/e5-small-v2                   | 384 |

For maze environments with a finite set of goal positions (e.g. ant-maze-u4
with 10 goals), embeddings are **precomputed once** at init and looked up
via nearest-neighbour argmin during training — the model is never run in the
training loop (saves ~60\% training time).

### Trainable embedding (``--trainable-embedding``)

Replaces the frozen text model with a learned ``nn.Embed`` lookup table.
Each discrete goal position gets its own trainable vector, optimised
end-to-end with the critic objective. Only works when the environment
provides a finite goal set (``possible_goals``).

### Hybrid encoder (``--hybrid-goal-encoder``)

Requires ``--text-encoder true``. Computes a text embedding (frozen or
trainable) and **combines it with the raw (x, y) coordinates** before
passing through a processing backbone. The backbone is selected
automatically based on ``--transformer-mode``:

**Dimension flow (frozen text model — 384-dim BERT → projected):**

| Backbone | Embed source | Text encoder ``output_dim`` | Concatenation | Hidden dim | Output dim |
|----------|-------------|----------------------------|---------------|------------|------------|
| MLP | Frozen | ``mlp_width`` (256) | ``g_proj(64) + text_repr(256) → 320`` | 256 | 64 |
| MLP | Trainable | ``mlp_width`` (256) | ``g_proj(64) + text_repr(256) → 320`` | 256 | 64 |
| Transformer | Frozen | ``transformer_embed_dim`` (144) | stacked as 2 tokens ← ``g_token(144), text_repr(144)`` | 144 | 64 |
| Transformer | Trainable | ``transformer_embed_dim`` (144) | stacked as 2 tokens ← ``g_token(144), text_repr(144)`` | 144 | 64 |

**Frozen path** (``--text-encoder true --trainable-embedding false``):
The BERT model outputs 384-dim embeddings which are projected to
``output_dim`` (matching the backbone width) via a learned ``nn.Dense``.
This avoids a 384→64→256 information bottleneck in the MLP case.

**Trainable path** (``--trainable-embedding true``):
An ``nn.Embed`` lookup table produces vectors of size ``output_dim``
directly — no intermediate 384-dim representation. The dimension matches
the backbone width so both paths are comparable.

**MLP backbone** (default or ``--transformer-mode none``):
Raw coords ``(x, y)`` are first projected to 64 dims (``g_proj``) so they
don't get drowned by the high-dim text representation. The projected coords
and the text embedding are concatenated and processed by a 2-layer MLP:
``Dense(mlp_width) → swish → Dense(64)``.

**Semantic transformer backbone** (``--transformer-mode StateGoal``
or ``Full``):
Raw coords are projected to ``transformer_embed_dim`` via ``Dense`` and
**stacked as separate tokens** alongside the text embedding, giving the
``TransformerBackbone`` two (or ``1 + seq_len``) tokens to attend to.

Pooling is controlled by ``--text-pooling``:

- **``cls``** (default): single CLS vector per goal — 2 tokens.
- **``mean``**: mean over all non-padding tokens — 2 tokens.
- **``token``**: all BERT token vectors kept — ``1 + seq_len`` tokens in
  the transformer backbone. The MLP backbone mean-pools to a single vector.

The text embedding source is controlled by ``--trainable-embedding``:

    # Raw coords + frozen MiniLM → MLP backbone
    --text-encoder true --hybrid-goal-encoder true

    # Raw coords + trainable embedding → semantic transformer
    --text-encoder true --hybrid-goal-encoder true \
        --trainable-embedding true --transformer-mode StateGoal
