#!/usr/bin/env python3
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: logging.py
@time: 2025/4/27 00:14
@desc: Logging module for the project
'''
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_norm = CURRENT_DIR.replace('\\', '/')
SRC_DIR = _norm[:_norm.find('/src/') + 4] if '/src/' in _norm else CURRENT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from utils.config import load_env_file
import builtins
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_OUTPUT_DIR = './'
LOG_FILE = 'scraper.log'
DEFAULT_FILE_SIZE = 1000  # 1000MB

# Add custom TIME logging level
# TIME level is between DEBUG (10) and INFO (20), set to 15
TIME_LEVEL = 15
logging.addLevelName(TIME_LEVEL, 'TIME')

COLORS = {
    'DEBUG': '\033[94m',        # Blue
    'INFO': '\033[97m',         # White (default)
    'WARNING': '\033[93m',      # Yellow
    'ERROR': '\033[91m',         # Red
    'CRITICAL': '\033[41;1m',   # Red background with bold white text
    'RESET': '\033[0m',          # Reset all attributes
    'TIME': '\033[92m'          # Green
}

# Unified level mapping to avoid duplication
LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'TIME': TIME_LEVEL,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

load_env_file()

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter"""
    def format(self, record):
        levelname = record.levelname
        msg = super().format(record)
        color = COLORS.get(levelname, COLORS['RESET'])
        reset = COLORS['RESET']
        return f"{color}{msg}{reset}"

class NoiseFilter(logging.Filter):
    """Filter out noisy debug logs (e.g., inotify events)"""
    NOISE_PATTERNS = [
        'in-event',
        'InotifyEvent',
        'IN_ISDIR',
        'IN_OPEN',
        '__pycache__',
    ]
    
    def filter(self, record):
        """Filter out noisy log messages"""
        message = record.getMessage()
        # Filter out logs containing noise patterns
        for pattern in self.NOISE_PATTERNS:
            if pattern.lower() in message.lower():
                return False
        return True

class Logger:
    """Logger class for managing logging configuration"""
    
    def __init__(self, log_level=None):
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.log_file = os.path.join(self.output_dir, LOG_FILE)
        self.log_level = log_level
        self._setup_logging()
        
    def set_output_dir(self, output_dir):
        """Set output directory for logs"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, LOG_FILE)
        self._setup_logging()
    
    def set_log_level(self, level):
        """Set log level manually"""
        if isinstance(level, str):
            level = LEVEL_MAP.get(level.upper(), logging.INFO)
        self.log_level = level
        self._setup_logging()
        
    def _get_default_log_level(self):
        """Get default log level with automatic detection"""
        # Priority 1: Manual setting (highest priority - explicit user setting)
        if self.log_level is not None:
            return self.log_level
        
        # Priority 2: LOG_LEVEL from .env or environment variable
        env_level = os.environ.get('LOG_LEVEL', '').upper()
        if env_level and env_level in LEVEL_MAP:
            return LEVEL_MAP[env_level]
        
        # Priority 3: RUN_MODE environment variable
        run_mode = os.environ.get('RUN_MODE', '').upper()
        if run_mode in ('PRODUCTION', 'RUN', 'PROD'):
            return logging.WARNING
        
        # Priority 4: Automatic detection
        # Debugger attached (keep INFO for debugging)
        if sys.gettrace() is not None:
            return logging.INFO
        
        # Test mode (only if no manual setting)
        if 'pytest' in sys.argv[0] or any('pytest' in arg for arg in sys.argv) or 'unittest' in sys.modules:
            return logging.WARNING
        
        # Run mode (python -O)
        if not __debug__:
            return logging.WARNING
        
        # Default: WARNING (when no .env or environment variable set)
        return logging.WARNING
        
    def _setup_logging(self):
        """Setup logging configuration"""
        # Get log level
        log_level = self._get_default_log_level()
        
        # Ensure output directory exists before creating file handler
        output_dir = os.path.dirname(self.log_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create console handler
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter('%(asctime)s [%(levelname)s] %(message)s'))
        console.setLevel(log_level)
        # Add noise filter to console handler
        console.addFilter(NoiseFilter())

        # Create file handler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=DEFAULT_FILE_SIZE*1024*1024,  # XX MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        # File handler always logs from DEBUG level to capture all logs
        file_handler.setLevel(logging.DEBUG)
        # Add noise filter to file handler as well
        file_handler.addFilter(NoiseFilter())

        # Configure root logger
        # Set root logger to DEBUG to allow file handler to capture all logs
        # Individual handlers will filter by their own levels
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        if logger.handlers:
            logger.handlers = []

        # Add handlers
        logger.addHandler(console)
        logger.addHandler(file_handler)

# Create global logger manager instance
logger_manager = Logger()


def _get_module_path(file_path):
    """Get module path from file path, relative to src directory"""
    normalized_path = file_path.replace('\\', '/')
    
    if '/src/' in normalized_path:
        src_index = normalized_path.find('/src/') + 5
        module_path = normalized_path[src_index:]
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        return module_path.replace('/', '.')
    else:
        return os.path.basename(file_path)


def _extract_caller_info(frame_number):
    """Extract caller information from frame
    
    Args:
        frame_number: Frame number relative to print_to_logging
                      (1 = caller of print_to_logging, 2 = caller's caller)
                      This function adds 2 to skip its own frame and _get_caller_string's frame
    
    Returns:
        str: Formatted caller info like "[module:function:line]"
    """
    try:
        # Add 2 to skip: frame 0 = _extract_caller_info, frame 1 = _get_caller_string
        # So frame_number=1 becomes frame 3 (print_to_logging's caller)
        # And frame_number=2 becomes frame 4 (caller's caller, for TIME level)
        frame = sys._getframe(frame_number + 2)
        func_name = frame.f_code.co_name
        file_path = frame.f_globals.get('__file__', '')
        line_number = frame.f_lineno
        module_path = _get_module_path(file_path)
        return f"[{module_path}:{func_name}:{line_number}] "
    except ValueError:
        return "[unknown:unknown:0] "


def _get_caller_string(log_level):
    """Get caller string based on log level
    
    Args:
        log_level: Log level (string or int)
    
    Returns:
        str: Formatted caller info
    """
    is_time_level = (log_level == "TIME" or log_level == TIME_LEVEL)
    
    if is_time_level:
        # For TIME level: skip print_to_logging (frame 0) and timeit wrapper (frame 1)
        # Need frame 2 relative to print_to_logging
        return _extract_caller_info(2)
    else:
        # For other levels: skip print_to_logging (frame 0)
        # Need frame 1 relative to print_to_logging
        return _extract_caller_info(1)


def _log_with_level(message, log_level, logger):
    """Log message with specified level
    
    Args:
        message: Message to log
        log_level: Log level (string or int)
        logger: Logger instance
    """
    if log_level == "INFO" or log_level == logging.INFO:
        logging.info(message)
    elif log_level == "WARNING" or log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == "ERROR" or log_level == logging.ERROR:
        logging.error(message)
    elif log_level == "CRITICAL" or log_level == logging.CRITICAL:
        logging.critical(message)
    elif log_level == "TIME" or log_level == TIME_LEVEL:
        logger.log(TIME_LEVEL, message)
    else:
        logging.debug(message)


def print_to_logging(*args, level="debug", **kwargs):
    """Custom print function that logs messages"""
    for k in ['end', 'flush']:
        kwargs.pop(k, None)
    
    log_level = level.upper() if isinstance(level, str) else level
    caller = _get_caller_string(log_level)
    message = caller + ' '.join(str(arg) for arg in args)
    
    logger = logging.getLogger()
    _log_with_level(message, log_level, logger)

def init_logger(output_dir, log_level=None):
    """Initialize logger with custom output directory and optional log level"""
    if log_level is not None:
        logger_manager.set_log_level(log_level)
    logger_manager.set_output_dir(output_dir)

def set_log_level(level):
    """Set log level for the global logger
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING, etc.)
              or string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TIME')
    """
    if isinstance(level, str):
        level = LEVEL_MAP.get(level.upper(), logging.INFO)
    logger_manager.set_log_level(level)


# replace print
builtins.print = print_to_logging

def main():
    """Test function to demonstrate different log levels"""
    print("This is a DEBUG message", level="DEBUG")
    print("This is an INFO message", level="INFO")
    print("This is a WARNING message", level="WARNING")
    print("This is an ERROR message", level="ERROR")
    print("This is a CRITICAL message", level="CRITICAL")
    print("This is a TIME message", level="TIME")

    # Show current log level
    current_level = logging.getLogger().level
    level_names = {
        logging.DEBUG: 'DEBUG',
        TIME_LEVEL: 'TIME',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARNING',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRITICAL'
    }
    print(f"Current log level: {level_names.get(current_level, current_level)}")

if __name__ == '__main__':
    main()
