__version__ = "0.0.4"


try:
    from sorted_nearest.src.sorted_nearest import (nearest, nearest_nonoverlapping,
                            nearest_previous_nonoverlapping, nearest_next_nonoverlapping, nearest_next,
                            nearest_previous)
except ImportError:
    pass
