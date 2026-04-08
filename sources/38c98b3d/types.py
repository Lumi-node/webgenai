"""Exception types and error hierarchy for design-to-HTML converter.

This module defines a complete exception hierarchy used across all conversion modules.
It serves as the foundational module with zero project dependencies (stdlib only).
"""


class DesignToHTMLError(Exception):
    """Base exception for all design-to-HTML conversion errors.

    This is the top-level exception that all conversion-specific errors inherit from.
    Use this to catch any error from the design-to-HTML conversion pipeline.
    """
    pass


class ImageLoadError(DesignToHTMLError):
    """Raised when image loading fails.

    This exception is raised when:
    - The image file is not found
    - The image format is not PNG or JPG
    - Image dimensions are outside the valid range [200-2000]px
    - The image file is corrupted or cannot be decoded
    """
    pass


class LayoutDetectionError(DesignToHTMLError):
    """Raised when layout detection fails.

    This exception is raised when the algorithm to detect header, sidebar, content,
    and footer regions encounters an error. This is rare but can occur if the input
    image data is invalid or the detection process encounters unexpected conditions.
    """
    pass


class ColorExtractionError(DesignToHTMLError):
    """Raised when color extraction fails.

    This exception is raised when the k-means clustering algorithm fails to extract
    dominant colors from detected regions. This can occur if the clustering process
    encounters invalid data or numerical errors.
    """
    pass


class HTMLGenerationError(DesignToHTMLError):
    """Raised when HTML or CSS generation fails.

    This exception is raised when the HTML structure or CSS styling generation fails.
    This can occur if the region data is invalid or the generation process encounters
    unexpected conditions.
    """
    pass


class OutputWriteError(DesignToHTMLError):
    """Raised when writing output files fails.

    This exception is raised when:
    - HTML validation fails (invalid HTML structure)
    - The output directory cannot be created
    - The index.html file cannot be written to disk
    - File I/O operations encounter errors
    """
    pass
