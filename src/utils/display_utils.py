"""
Display utility functions.

This module provides utilities for formatting console output.
"""

def show_divider():
    """
    Print a visual divider line for console output formatting.
    
    Prints a line of 20 equal signs to create visual separation
    in console output for better readability.
    """
    print("=" * 20)


def show_with_start_divider(content):
    """
    Display content with a divider line at the start.
    
    Args:
        content (str): Content to display after the divider
    """
    show_divider()
    print(content)


def show_with_end_divider(content):
    """
    Display content with a divider line at the end.
    
    Args:
        content (str): Content to display before the divider
    """
    print(content)
    show_divider()
    print()
