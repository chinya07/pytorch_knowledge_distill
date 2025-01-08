"""Models module."""

from .deep_model import DeepNN
from .light_model import LightNN
from .modified_deep_model import ModifiedDeepNN
from .modified_light_model import ModifiedLightNN

__all__ = ['DeepNN', 'ModifiedDeepNN', 'LightNN', 'ModifiedLightNN'] 