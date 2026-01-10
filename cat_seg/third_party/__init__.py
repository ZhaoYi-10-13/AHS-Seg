# Third party modules
from .clip import load, tokenize
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

# Enhanced OFT modules for ECCV 2026
from .adaptive_energy import (
    AdaptiveEnergyModulator,
    AdaptiveOFTLayer,
    energy_tracker,
    GlobalEnergyTracker
)
from .enhanced_oft import (
    EnhancedOFTLinearLayer,
    EnhancedOFTQKVLayer,
    EnhancedOFTOutputLayer,
    set_enhanced_oft,
    set_enhanced_oft_vision
)
