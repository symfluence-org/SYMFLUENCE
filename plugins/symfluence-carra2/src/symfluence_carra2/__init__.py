"""Optional CARRA2 forcing plugin. Install alongside SYMFLUENCE."""
def register():
    from symfluence.core.registries import R
    from .acquisition import CARRA2Acquirer
    from .dataset import CARRA2Handler
    R.acquisition_handlers.add('CARRA2', CARRA2Acquirer)
    R.dataset_handlers.add('carra2', CARRA2Handler)
