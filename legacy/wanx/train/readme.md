# Wan-Video-with-Sparsity Usage Guide

## 1. Basic Usage
```python
from modify_wan import set_adaptive_block_sparse_attn_wanx  
set_adaptive_block_sparse_attn_wanx(pipe.transformer)
```
`wan.py` is a good example to refer to.

## 2. Configuration

### 2.1 Training
- **File**: `./special_attentions_local/TrainRelated/blocksparseattn.py`
- **Modification**: 
  - Set ` depth` to match the training configuration.
  - Adjust other configurations as needed.
- **Note**: Comment out print statements during training to avoid unnecessary output.

### 2.2 Inference
- Set `depth` to 21.
- Use the same parameters as those specified in the training configuration.