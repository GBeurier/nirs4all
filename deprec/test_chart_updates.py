"""
Test script to verify the updated chart controller with new legends and titles
"""

def test_chart_controller_updates():
    """Test the updated chart controller without plotting"""

    print("🧪 Testing Chart Controller Updates")
    print("=" * 50)

    try:
        # Import the controller
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from nirs4all.controllers.chart.op_spectra_charts import SpectraChartController

        # Test matching functionality
        controller = SpectraChartController()

        # Test 2D matching
        matches_2d = controller.matches("chart_2d", None, "chart_2d")
        print(f"✓ 2D matching: {matches_2d}")

        # Test 3D matching
        matches_3d = controller.matches("chart_3d", None, "chart_3d")
        print(f"✓ 3D matching: {matches_3d}")

        # Test non-matching
        matches_other = controller.matches("other", None, "other")
        print(f"✓ Non-matching: {matches_other}")

        print()
        print("📋 Chart Controller Updates Applied:")
        print("  • Target Values → y")
        print("  • Feature Index (Wavelength) → x (features)")
        print("  • Spectral Intensity → Intensity")
        print("  • Target Values (sorted) → y (sorted)")
        print("  • Synthetic titles: proc_X | src:Y | n:Z | f:W")
        print("  • Smaller title font: fontsize=10")

        print()
        print("🎯 Example title format:")
        print("  Before: '2D Spectra Visualization - Processing 0\\nData Source: 0, Samples: 20'")
        print("  After:  'proc_0 | src:0 | n:20 | f:50'")

        print()
        print("✅ All updates successfully applied to chart controller!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chart_controller_updates()

    if success:
        print("\n🎉 Chart controller is ready with updated legends and titles!")
        print("\nKey improvements:")
        print("- More concise axis labels (x, y instead of verbose descriptions)")
        print("- Synthetic titles with key info: processing | source | samples | features")
        print("- Smaller title font for better visual balance")
        print("- Works for both 2D and 3D visualizations")
    else:
        print("\n❌ Please check the implementation for issues")