from loguru import logger
from new_modeling_toolkit.core.temporal.new_temporal import NewTemporalSettings
from new_modeling_toolkit.resolve.settings import CustomConstraints

def validate_resolve_settings(temporal_settings: NewTemporalSettings, custom_constraints: CustomConstraints):
    """Validate resolve settings before running the model.
    
    Args:
        temporal_settings: The temporal settings to validate
        custom_constraints: The custom constraints to validate
    """
    logger.info("Validating resolve settings...")
    
    # Validate temporal settings
    if temporal_settings is None:
        raise ValueError("Temporal settings cannot be None")
        
    # Validate custom constraints
    if custom_constraints is None:
        logger.warning("No custom constraints found")
    
    logger.info("Settings validation complete") 