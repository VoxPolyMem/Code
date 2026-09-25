"""Isolated AudioMem retrieval adapter v2.

Nothing in this package is imported by the frozen Mem-Gallery, H2HMem, or
shared r12 paths.  The feature remains explicit and default-off.
"""

from .retrieval_adapter import relation_profile_context, raw_hybrid_topk

__all__ = ["relation_profile_context", "raw_hybrid_topk"]
