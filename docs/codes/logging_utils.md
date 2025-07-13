# Logging Utilities

The `logging_utils.py` module provides utilities for logging and progress tracking in the project.

## Overview

This module includes functions for setting up structured logging with the `loguru` library and creating progress bars with the `rich` library. It also provides a decorator for wrapping functions with logging.

## Key Components

### Logger Setup

```python
def setup_logger(log_file):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(log_file, rotation="10 MB")
```

This function configures the logger with:
- A colorized console output format
- File logging with automatic rotation at 10 MB

### Task Wrapper Decorator

```python
def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper
```

This decorator:
- Logs when a function starts and finishes
- Catches and logs any exceptions that occur
- Preserves the original function's metadata using `@wraps`

### Rich Progress Bar

```python
def get_rich_progress():
    return Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True)
```

This function creates a customized progress bar with:
- A spinner animation
- A text description column
- Transient display (clears after completion)

## Usage

### Logger Setup

```python
from utils.logging_utils import setup_logger

# Set up logging to a file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
setup_logger(log_dir / "my_log.log")
```

### Task Wrapper

```python
from utils.logging_utils import task_wrapper

@task_wrapper
def my_function():
    # Function code here
    pass
```

### Progress Bar

```python
from utils.logging_utils import get_rich_progress

with get_rich_progress() as progress:
    task = progress.add_task("[green]Processing...", total=100)
    
    for i in range(100):
        # Do some work
        progress.advance(task)
```

## Code Reference

```1:4:src/utils/logging_utils.py
import sys
from functools import wraps

from loguru import logger
```

```7:12:src/utils/logging_utils.py
def setup_logger(log_file):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(log_file, rotation="10 MB")
```

```15:25:src/utils/logging_utils.py
def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper
``` 