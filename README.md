# A Keras 3 translation of FocalNet
<a target="_blank" href="https://colab.research.google.com/github/anas-rz/focalnet-keras-core/blob/main/colab_usage.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Released by Microsoft in 2022, FocalNet or Focal Modulation Network is an attention-free architecture achieving superior performance than SoTA self-attention (SA) methods across various vision benchmarks.  [GitHub](https://github.com/microsoft/FocalNet/) [Paper](https://arxiv.org/abs/2203.11926).


# Installation



```
git clone https://github.com/anas-rz/focalnet-keras-3.git
cd focalnet-keras-3
```

# Usage

```
from focalnet_keras_core import *
model = focalnet_huge_fl3()

```

# Available Functions:


```
focalnet_tiny_srf

focalnet_small_srf

focalnet_base_srf

focalnet_tiny_lrf

focalnet_small_lrf

focalnet_base_lrf

focalnet_tiny_iso_16

focalnet_small_iso_16

focalnet_base_iso_16

focalnet_large_fl3

focalnet_large_fl4

focalnet_xlarge_fl3

focalnet_xlarge_fl4

focalnet_huge_fl3

focalnet_huge_fl4
```
