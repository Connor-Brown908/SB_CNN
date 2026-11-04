"""seqparam: predict E. coli circuit parameters from DNA sequence (Stage-1 CNNs).

Two parameters, two models, one shared pipeline:
  - promoter strength / transcription rate   (element_type="promoter")
  - RBS / translation rate                   (element_type="rbs")
"""

__version__ = "0.1.0"

ALPHABET = "ACGT"
ELEMENT_TYPES = ("promoter", "rbs")
