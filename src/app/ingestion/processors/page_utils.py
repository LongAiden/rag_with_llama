"""
Utility functions for page number mapping.
Used by all document processors to track which page each chunk belongs to.
"""

def get_page_number_for_position(position: int, page_mapping: list) -> int:
    """
    Get page number for a given text position.
    
    Args:
        position: Character position in text
        page_mapping: List of (start_pos, end_pos, page_num) tuples
    
    Returns:
        int: Page number (1-indexed)
    
    Example:
        >>> page_mapping = [(0, 100, 1), (101, 200, 2)]
        >>> get_page_number_for_position(150, page_mapping)
        2
    """
    for start_pos, end_pos, page_num in page_mapping:
        if start_pos <= position <= end_pos:
            return page_num
    
    # If not found, estimate based on closest page
    if page_mapping:
        for start_pos, end_pos, page_num in page_mapping:
            if position < start_pos:
                return page_num
        # If position is after all mapped content, return last page
        return page_mapping[-1][2]
    
    return 1  # Default to page 1



def get_supported_file_types():
    """
    Get all supported file types across all registered processors.

    Returns:
        list: List of supported file extensions (e.g., ['.pdf', '.docx', '.txt'])
    """
    from app.ingestion.processors.processor_factory import get_registry

    registry = get_registry()
    return registry.get_supported_extensions()


def list_available_processors():
    """
    List all available document processors.

    Returns:
        list: List of registered DocumentProcessor instances
    """
    from app.ingestion.processors.processor_factory import get_registry

    registry = get_registry()
    return registry.list_processors()

