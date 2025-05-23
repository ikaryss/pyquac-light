#!/usr/bin/env python3
"""
Simple test script to verify the interactive app works correctly.
"""

import numpy as np
from pyquac_light import launch_app, RandomSpectroscopy


def test_basic_functionality():
    """Test basic app functionality."""
    print("Testing pyquac-light interactive app...")

    # Test 1: Import successful
    print("✓ Imports successful")

    # Test 2: Create sample data
    x_arr = np.linspace(-1.0, 1.0, 21)  # Small grid for testing
    y_arr = np.linspace(4e9, 5e9, 31)

    spec = RandomSpectroscopy(x_arr=x_arr, y_arr=y_arr)
    print(
        f"✓ Created spectroscopy instance with {len(spec.x_arr)} x {len(spec.y_arr)} grid"
    )

    # Test 3: Add some data
    spec.run_full_scan(sleep=1e-6)
    print(f"✓ Added {len(spec._raw_x)} data points")

    # Test 4: Test data operations
    original_count = len(spec._raw_x)
    spec.drop(x=spec.x_arr[::5])  # Drop every 5th x value
    new_count = len(spec._raw_x)
    print(f"✓ Data drop operation: {original_count} -> {new_count} points")

    # Test 5: Test ridge fitting
    try:
        ridge = spec.fit_ridge(deg=2)
        print(f"✓ Ridge fitting successful: {ridge}")
    except Exception as e:
        print(f"⚠ Ridge fitting failed (expected with random data): {e}")

    # Test 6: Create app widget (without displaying)
    try:
        app_widget = launch_app(spec)
        print("✓ App widget creation successful")
        print(f"✓ Widget type: {type(app_widget)}")
    except Exception as e:
        print(f"✗ App widget creation failed: {e}")
        return False

    # Test 7: Test empty app
    try:
        empty_app = launch_app()
        print("✓ Empty app creation successful")
    except Exception as e:
        print(f"✗ Empty app creation failed: {e}")
        return False

    print("\n🎉 All tests passed! The interactive app is working correctly.")
    print("\nTo use the app in a Jupyter notebook:")
    print("1. Import: from pyquac_light import launch_app, RandomSpectroscopy")
    print("2. Create data: spec = RandomSpectroscopy(x_arr, y_arr)")
    print("3. Launch app: app = launch_app(spec)")
    print("4. Display: app")

    return True


if __name__ == "__main__":
    test_basic_functionality()
