"""
Demo of Clean Pipeline Architecture

Shows the separation of concerns:
- Runner: pure execution with context
- Builder: generic building from any format
- Operations: simple wrappers
"""

# Mock classes for demonstration
class MockDataset:
    def __init__(self):
        self.data = "mock_data"
    def select(self, **filters):
        return self
    def get_features(self):
        return [[1, 2, 3], [4, 5, 6]]

class MockContext:
    def __init__(self):
        self.current_filters = {}

class MockOperation:
    def __init__(self, name, operator=None):
        self.name = name
        self.operator = operator
    def execute(self, dataset, context):
        print(f"    Executing {self.name}")
    def get_name(self):
        return self.name

class MockBuilder:
    def build_operation(self, step):
        if isinstance(step, str):
            return MockOperation(f"StringOp({step})")
        elif isinstance(step, dict):
            return MockOperation(f"DictOp({list(step.keys())})")
        else:
            return MockOperation(f"ObjectOp({type(step).__name__})")

# Clean Runner Implementation
class CleanPipelineRunner:
    def __init__(self):
        self.builder = MockBuilder()
        self.context = MockContext()
        self.step_count = 0

    def run_pipeline(self, steps):
        print("🚀 Clean Pipeline Runner")
        for step in steps:
            self._run_step(step)

    def _run_step(self, step, prefix=""):
        """MAIN PARSING LOGIC - Clean and transparent"""
        self.step_count += 1

        print(f"{prefix}📋 Step {self.step_count}: {self._describe_step(step)}")

        # CONTROL OPERATIONS - handled locally
        if isinstance(step, str) and step == "uncluster":
            self._run_uncluster(prefix + "  ")
        elif isinstance(step, str) and step.startswith("Plot"):
            self._run_visualization(step, prefix + "  ")
        elif isinstance(step, list):
            self._run_sub_pipeline(step, prefix + "  ")
        elif isinstance(step, dict) and len(step) == 1:
            key = next(iter(step.keys()))
            if key in ["dispatch", "sample_augmentation", "feature_augmentation"]:
                print(f"{prefix}  🔄 Control operation: {key}")
            else:
                # DATA OPERATION - delegate to builder
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, prefix + "  ")
        else:
            # DATA OPERATION - delegate to builder
            operation = self.builder.build_operation(step)
            self._execute_operation(operation, prefix + "  ")

    def _execute_operation(self, operation, prefix=""):
        """Simple select and execute"""
        print(f"{prefix}🔧 {operation.get_name()}")
        dataset = MockDataset()
        operation.execute(dataset, self.context)

    def _run_uncluster(self, prefix):
        print(f"{prefix}🔓 Uncluster control")

    def _run_visualization(self, step, prefix):
        print(f"{prefix}📊 Visualization: {step}")

    def _run_sub_pipeline(self, steps, prefix):
        print(f"{prefix}📋 Sub-pipeline ({len(steps)} steps)")
        for sub_step in steps:
            self._run_step(sub_step, prefix + "  ")

    def _describe_step(self, step):
        if isinstance(step, str):
            return f"'{step}'"
        elif isinstance(step, dict):
            return f"dict {list(step.keys())}"
        elif isinstance(step, list):
            return f"list ({len(step)} items)"
        else:
            return f"{type(step).__name__}"


# Demo the clean architecture
def demo_clean_pipeline():
    """Demo showing the clean separation of concerns"""

    print("=" * 60)
    print("CLEAN PIPELINE ARCHITECTURE DEMO")
    print("=" * 60)

    # Sample pipeline config with different step types
    pipeline_config = [
        # String preset
        "StandardScaler",

        # Control operation
        {"cluster": "KMeans"},

        # Sub-pipeline
        ["PCA", "MinMaxScaler"],

        # Control commands
        "uncluster",
        "PlotData",

        # Complex dict
        {"sample_augmentation": ["NoiseAugmenter", "RotationAugmenter"]},

        # Generic operation dict
        {"class": "sklearn.decomposition.PCA", "params": {"n_components": 2}},
    ]

    runner = CleanPipelineRunner()
    runner.run_pipeline(pipeline_config)

    print("\n" + "=" * 60)
    print("ARCHITECTURE BENEFITS:")
    print("✅ Runner: Pure execution logic, clear config parsing")
    print("✅ Builder: Generic building, handles all formats")
    print("✅ Operations: Simple wrappers, focused responsibility")
    print("✅ Serialization: Handled by builder, reusable across branches")
    print("✅ Context: Explicit filtering and state management")
    print("=" * 60)


if __name__ == "__main__":
    demo_clean_pipeline()
