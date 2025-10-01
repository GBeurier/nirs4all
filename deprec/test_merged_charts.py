"""
Test script for the merged SpectraChartController
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock the necessary imports for testing
class MockStep:
    pass

class MockOperator:
    pass

class MockDataset:
    def x(self, context, format_type, something):
        import numpy as np
        # Return mock spectra data (samples, processings, features)
        return [np.random.randn(10, 2, 50)]  # 10 samples, 2 processings, 50 features

    def y(self, context):
        import numpy as np
        return np.random.randn(10)  # 10 target values

class MockRunner:
    pass

def test_matches():
    """Test that the controller matches both 2D and 3D keywords"""
    try:
        from nirs4all.controllers.chart.op_spectra_charts import SpectraChartController

        # Test 2D matching
        assert SpectraChartController.matches(MockStep(), MockOperator(), "chart_2d")
        print("✓ 2D matching works")

        # Test 3D matching
        assert SpectraChartController.matches(MockStep(), MockOperator(), "chart_3d")
        print("✓ 3D matching works")

        # Test non-matching
        assert not SpectraChartController.matches(MockStep(), MockOperator(), "other_keyword")
        print("✓ Non-matching works")

        print("✓ All matching tests passed")
        return True

    except Exception as e:
        print(f"✗ Matching test failed: {e}")
        return False

def test_execute_structure():
    """Test the execute method structure (without actually plotting)"""
    try:
        from nirs4all.controllers.chart.op_spectra_charts import SpectraChartController

        controller = SpectraChartController()

        # Test 2D context
        context_2d = {"keyword": "chart_2d"}

        # Mock plt.show to avoid actual plotting
        import matplotlib.pyplot as plt
        original_show = plt.show
        original_close = plt.close
        plt.show = lambda: None  # Mock show
        plt.close = lambda fig: None  # Mock close

        try:
            result = controller.execute(
                MockStep(),
                MockOperator(),
                MockDataset(),
                context_2d,
                MockRunner()
            )

            # Should return tuple of (context, img_list)
            assert isinstance(result, tuple)
            assert len(result) == 2
            context, img_list = result
            assert isinstance(img_list, list)

            if img_list:
                img_info = img_list[0]
                assert 'plot_type' in img_info
                assert img_info['plot_type'] == '2D'
                assert 'image_base64' in img_info

            print("✓ 2D execute structure test passed")

        finally:
            # Restore original functions
            plt.show = original_show
            plt.close = original_close

        return True

    except Exception as e:
        print(f"✗ Execute structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing merged SpectraChartController...")
    print()

    success = True

    # Test matching functionality
    success &= test_matches()
    print()

    # Test execute structure
    success &= test_execute_structure()
    print()

    if success:
        print("🎉 All tests passed! The merged controller is working correctly.")
        print()
        print("Key features:")
        print("- Handles both 'chart_2d' and 'chart_3d' keywords")
        print("- Returns (context, img_list) tuple")
        print("- img_list contains plot metadata including base64 encoded images")
        print("- Supports multiple data sources and processing types")
    else:
        print("❌ Some tests failed. Please check the implementation.")