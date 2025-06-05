class Pipeline:
    """
    A class to represent a pipeline for processing data.
    """

    def __init__(self, name: str = "Default Pipeline"):
        """
        Initializes the Pipeline with a name.

        :param name: The name of the pipeline.
        """
        self.name = name
        self.steps = []

    # def add_step(self, step):
    #     """
    #     Adds a processing step to the pipeline.

    #     :param step: The processing step to add.
    #     """
    #     self.steps.append(step)

    # def run(self, data):
    #     """
    #     Runs the pipeline on the given data.

    #     :param data: The data to process.
    #     :return: The processed data.
    #     """
    #     for step in self.steps:
    #         data = step.process(data)
    #     return data