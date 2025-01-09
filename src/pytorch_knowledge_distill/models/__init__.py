"""Models module."""

from .deep_model import DeepNN
from .light_model import LightNN
from .cosine_embedding_deep_model import CosineEmbeddingDeepNN
from .cosine_embedding_light_model import CosineEmbeddingLightNN
from .modified_light_regressor_model import ModifiedLightRegressorNN    
from .modified_deep_regressor_model import ModifiedDeepRegressorNN

__all__ = ['DeepNN', 'CosineEmbeddingDeepNN', 'LightNN', 'CosineEmbeddingLightNN', 'ModifiedLightRegressorNN', 'ModifiedDeepRegressorNN'] 