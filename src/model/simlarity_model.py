from dataclasses import dataclass
from .similarity_interface import SimilarityInterface

@dataclass
class SimilarityModel:
    name: str 
    image_size: int
    model_cls: SimilarityInterface
    image_input_type: str = 'array'