from app.db.base_class import Base

# Import all models here to register them with Base
# This import must be at the end of the file to avoid circular imports
from app.models.user import User  # noqa
from app.models.signal import Signal  # noqa
from app.models.license import License  # noqa 