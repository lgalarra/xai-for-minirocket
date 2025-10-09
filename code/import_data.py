class DataImporter:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_path = f"data/{dataset_name}"

    def fetch_computed_attributions(self, instance_id, classifier, label, reference_policy,
                                    explainer_method, type='backpropagated'):
        pass