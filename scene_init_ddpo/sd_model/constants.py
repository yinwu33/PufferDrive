"""Constants vendored from scenario-dreamer ``cfgs/config.py``.

Only the values referenced by the diffusion *inference* closure are kept here so
the vendored model does not depend on scenario-dreamer's config package.
"""

# Partition flags (used by partition_mask inpainting logic in dm.py)
NON_PARTITIONED = 0
PARTITIONED = 1

AFTER_PARTITION = 0
BEFORE_PARTITION = 1
