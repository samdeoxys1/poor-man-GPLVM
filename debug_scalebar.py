#!/usr/bin/env python3
"""
Debug script to test the add_scalebar function
"""

import numpy as np
import matplotlib.pyplot as plt
from poor_man_gplvm.plot_helper import add_scalebar, add_scalebar_debug

def test_scalebar_basic():
    """Test basic scale bar functionality"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    print(f"X limits: {xlim}")
    print(f"Y limits: {ylim}")
    
    # Try adding a horizontal scale bar
    # Position it at bottom-right of the plot
    scale_x = xlim[1] - 2  # 2 units from right edge
    scale_y = ylim[0] + 0.1 * (ylim[1] - ylim[0])  # 10% from bottom
    scale_length = 1.0
    
    print(f"Adding scale bar at position ({scale_x}, {scale_y}) with length {scale_length}")
    
    add_scalebar_debug(ax, scale_x, scale_y, scale_length, 
                       label="1 unit", orientation='horizontal',
                       linewidth=3, color='red', zorder=10)
    
    # Try adding a vertical scale bar too
    scale_x_vert = xlim[0] + 0.1 * (xlim[1] - xlim[0])  # 10% from left
    scale_y_vert = ylim[0] + 0.1 * (ylim[1] - ylim[0])  # 10% from bottom
    scale_length_vert = 0.5
    
    print(f"Adding vertical scale bar at position ({scale_x_vert}, {scale_y_vert}) with length {scale_length_vert}")
    
    add_scalebar_debug(ax, scale_x_vert, scale_y_vert, scale_length_vert,
                       label="0.5 unit", orientation='vertical',
                       linewidth=3, color='blue', zorder=10)
    
    ax.set_title("Scale Bar Test")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    
    plt.tight_layout()
    plt.savefig('/Users/szheng/project/poor-man-GPLVM/scalebar_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def test_scalebar_edge_cases():
    """Test scale bar with different positioning scenarios"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Create different data for each subplot
        x = np.linspace(0, 5, 50)
        y = np.sin(x * (i + 1)) * (i + 1)
        ax.plot(x, y)
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Test different positions
        positions = [
            (xlim[1] - 1, ylim[0] + 0.1 * (ylim[1] - ylim[0])),  # bottom-right
            (xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0])),  # top-left
            (xlim[1] - 1, ylim[1] - 0.1 * (ylim[1] - ylim[0])),  # top-right
            (xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.1 * (ylim[1] - ylim[0]))   # bottom-left
        ]
        
        pos_x, pos_y = positions[i]
        
        print(f"Subplot {i}: Adding scale bar at ({pos_x:.2f}, {pos_y:.2f})")
        
        add_scalebar(ax, pos_x, pos_y, 0.5, 
                     label=f"0.5 unit", orientation='horizontal',
                     linewidth=2, color='red', zorder=10)
        
        ax.set_title(f"Subplot {i+1}")
    
    plt.tight_layout()
    plt.savefig('/Users/szheng/project/poor-man-GPLVM/scalebar_edge_cases.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes

if __name__ == "__main__":
    print("Testing basic scale bar functionality...")
    test_scalebar_basic()
    
    print("\nTesting scale bar edge cases...")
    test_scalebar_edge_cases()
    
    print("\nScale bar tests completed. Check the saved PNG files.")
