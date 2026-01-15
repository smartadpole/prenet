import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
_norm = CURRENT_DIR.replace('\\', '/')
SRC_DIR = _norm[:_norm.find('/src/') + 4] if '/src/' in _norm else CURRENT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



from .config import get_api_key, load_env_file, get_config, remove_bom, mask_secret
from review_core.utils import logger

__all__ = [
    'get_api_key', 'load_env_file', 'get_config',
    'remove_bom', 'mask_secret'
]
