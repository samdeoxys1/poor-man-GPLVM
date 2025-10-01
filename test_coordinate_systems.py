#!/usr/bin/env python3
"""
Test script to demonstrate both data and axes coordinate systems for scale bars
"""

import numpy as np
import matplotlib.pyplot as plt
from poor_man_gplvm.plot_helper import add_scalebar, add_scalebar_debug

def test_coordinate_systems():
    """Test both data and axes coordinate systems"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test data for all subplots
    x = np.linspace(0, 10, 100)
    y_data = [
        np.sin(x),
        np.cos(x) * 5,
        np.exp(-x/5) * np.sin(x*2),
        np.sin(x) * x
    ]
    
    titles = [
        "Data Coordinates (Original)",
        "Axes Coordinates (Fraction)",
        "Mixed: Data position, Axes length",
        "Comparison: Both systems"
    ]
    
    for i, (ax, y, title) in enumerate(zip(axes.flat, y_data, titles)):
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Data coordinates
            print(f"\n=== {title} ===")
            # Use data coordinates (original behavior)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Position scale bar in bottom-right
            scale_x = xlim[1] - 2
            scale_y = ylim[0] + 0.1 * (ylim[1] - ylim[0])
            
            add_scalebar_debug(ax, scale_x, scale_y, 1.0, 
                             label="1 unit (data)", orientation='horizontal',
                             color='red', linewidth=3, coord_system='data')
            
        elif i == 1:  # Axes coordinates
            print(f"\n=== {title} ===")
            # Use axes fraction coordinates (0-1)
            # Position scale bar at 70% right, 10% up from bottom
            add_scalebar_debug(ax, 0.7, 0.1, 0.2, 
                             label="20% width", orientation='horizontal',
                             color='green', linewidth=3, coord_system='axes')
            
            # Add a vertical scale bar too
            add_scalebar_debug(ax, 0.1, 0.1, 0.3,
                             label="30% height", orientation='vertical',
                             color='purple', linewidth=3, coord_system='axes')
            
        elif i == 2:  # Mixed example
            print(f"\n=== {title} ===")
            # Show how axes coordinates are more robust to different data ranges
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # This will always be in the same relative position regardless of data range
            add_scalebar_debug(ax, 0.6, 0.85, 0.25,
                             label="25% of axis width", orientation='horizontal',
                             color='orange', linewidth=3, coord_system='axes')
            
        else:  # Comparison
            print(f"\n=== {title} ===")
            # Show both systems on the same plot
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Data coordinates - position depends on actual data values
            scale_x_data = xlim[1] - 2
            scale_y_data = ylim[0] + 0.2 * (ylim[1] - ylim[0])
            add_scalebar(ax, scale_x_data, scale_y_data, 1.0,
                        label="1 unit (data)", orientation='horizontal',
                        color='red', linewidth=2, coord_system='data')
            
            # Axes coordinates - position is always relative to plot area
            add_scalebar(ax, 0.6, 0.8, 0.2,
                        label="20% width (axes)", orientation='horizontal',
                        color='blue', linewidth=2, coord_system='axes')
    
    plt.tight_layout()
    plt.savefig('/Users/szheng/project/poor-man-GPLVM/coordinate_systems_test.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes

def test_edge_cases():
    """Test edge cases and error handling"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(-5, 15, 100)
    y = np.sin(x) * np.exp(-x/10)
    ax.plot(x, y)
    ax.set_title("Edge Cases and Error Handling")
    
    print("\n=== Testing Edge Cases ===")
    
    # Test invalid coordinate system
    try:
        add_scalebar(ax, 0.5, 0.5, 0.1, coord_system='invalid')
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test axes coordinates with values outside 0-1
    print("\nTesting axes coordinates outside 0-1 range:")
    add_scalebar_debug(ax, 1.2, 0.5, 0.1,  # x > 1
                      label="Outside range", coord_system='axes',
                      color='red')
    
    # Test normal axes coordinates
    print("\nTesting normal axes coordinates:")
    add_scalebar_debug(ax, 0.7, 0.1, 0.2,
                      label="Normal axes", coord_system='axes',
                      color='green', linewidth=3)
    
    plt.tight_layout()
    plt.savefig('/Users/szheng/project/poor-man-GPLVM/edge_cases_test.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    print("Testing coordinate systems for scale bars...")
    test_coordinate_systems()
    
    print("\n" + "="*50)
    print("Testing edge cases...")
    test_edge_cases()
    
    print("\nTests completed! Check the saved PNG files.")
    print("\nUsage examples:")
    print("# Data coordinates (original behavior):")
    print("add_scalebar(ax, 5.0, 2.0, 1.0, label='1 unit', coord_system='data')")
    print("\n# Axes fraction coordinates (0-1):")
    print("add_scalebar(ax, 0.8, 0.1, 0.15, label='15% width', coord_system='axes')")
