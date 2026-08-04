class Estimator:
    def __init__(self, *args, **kwargs):
        self.lead_time = kwargs.get("lead_time", 0)
