from typing import Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
import importlib.util
from new_modeling_toolkit.core.component import Component
from new_modeling_toolkit.core.custom_model import Load

class System(BaseModel):
    # ...
    loads: Dict[str, Load] = Field(default_factory=dict)
    # ...
    @classmethod
    def from_csv(
        cls,
        folder: Union[str, Path],
        only_subfolders: Optional[List[str]] = None,
    ) -> "System":
        """Create a System instance from a folder of CSV files."""
        system = cls()
        for subfolder in sorted(Path(folder).iterdir()):
            if only_subfolders is not None and subfolder.name not in only_subfolders:
                continue

            if subfolder.is_dir():
                components = None
                # Check if there is a custom model for this subfolder
                module_name = f"new_modeling_toolkit.core.custom_model.{subfolder.name}"
                if importlib.util.find_spec(module_name) is not None:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "from_csv"):
                        components = getattr(module, "from_csv")(folder)

                if components is None:
                    components = Component.from_csv(folder, only_subfolders=[subfolder.name])

                for component in components.values():
                    if isinstance(component, Load):
                        system.loads[component.name] = component
                    # ... other component types ...

        return system

    @classmethod
    def from_dir(cls, settings_dir: Path):
        """Load a system from a directory."""
        # ... existing code ...