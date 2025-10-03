import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import arff
from urllib.request import urlretrieve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50  # type: ignore
from sklearn.utils import resample


class Data:
    def __init__(self, model_name=None, dataset_name=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_dir = "Data"
        self.image_data_loaded = False
        self.image_class_names = None

    def load_data(self, random_state=42):
        """Load and preprocess dataset."""
        data_loaders = {
            "german": self._load_german,
            "adult": self._load_adult,
            "bank": self._load_bank,

            "bike": self._load_bike,
            "compas": self._load_compas,

            "parkinson": self._load_parkinson,
            "thoraric": self._load_thoraric,
            "thyroid": self._load_thyroid,

            "image": self._load_image,
        }

        if self.dataset_name not in data_loaders:
            raise ValueError(f"Dataset '{self.dataset_name}' not found.")

        data = data_loaders[self.dataset_name]()

        if self.dataset_name == "image":
            X, Y, self.image_class_names = data
            self.image_data_loaded = True
            return X, Y
        
        else:
            X, Y = data

            return train_test_split(X, Y, test_size=0.2, random_state=random_state)

    # --- Utility Methods ---

    @staticmethod
    def _preprocess_data(X):
        """Apply standard scaling to features."""
        scaler = preprocessing.StandardScaler()
        return pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    @staticmethod
    def _encode_labels(Y):
        """Encode categorical labels."""
        encoder = preprocessing.LabelEncoder()
        return pd.Series(encoder.fit_transform(Y), index=Y.index)
    
    # --- Dataset Loaders ---

    def _load_adult(self):
        """Load and preprocess the Adult census dataset."""

        file_path = os.path.join(self.data_dir, "adult.data")
        columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num",
                   "Marital Status", "Occupation", "Relationship", "Race", "Sex",
                   "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"]
        data = pd.read_csv(file_path, names=columns, na_values="?")
        
        # Drop unused features
        data.drop(["Education", "fnlwgt"], axis=1, inplace=True)
        data["Target"] = (data["Target"] == " >50K").astype(int)

        cat_cols = ["Workclass", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
        num_cols = ["Age", "Capital Gain", "Capital Loss", "Hours per week"]
        
        for col in cat_cols:
            data[col] = self._encode_labels(data[col])        
        data[num_cols] = self._preprocess_data(data[num_cols])

        return data.drop("Target", axis=1), data["Target"]

    def _load_german(self):
        """Load and preprocess the German credit dataset."""

        file_path = os.path.join(self.data_dir, "german.data")
        columns = ["Payment_Status", "Duration_Monthly", "Credit_History", "Purpose", "Credit_Amount", "Savings_Account", "Current_Employment_Duration", "Installment_Rate", "Sex_Marital_Status", "Guarantors", "Current_Address_Duration", "Property", "Age", "Other_Installment_Plans", "Housing", "Num_Existing_Credits", "Occupation", "Num_Dependents", "Telephone", "Foreign_Worker", "Target"]

        data = pd.read_csv(file_path, names=columns, sep=" ")

        cat_cols = ["Payment_Status", "Credit_History",  "Purpose", "Savings_Account", "Current_Employment_Duration", "Sex_Marital_Status", "Guarantors", "Property", "Other_Installment_Plans", "Housing", "Occupation", "Telephone", "Foreign_Worker", "Target"]
        num_cols = ["Duration_Monthly", "Credit_Amount", "Installment_Rate", "Current_Address_Duration", "Age", "Num_Existing_Credits", "Num_Dependents" ]

        for col in cat_cols:
            data[col] = self._encode_labels(data[col])        
        data[num_cols] = self._preprocess_data(data[num_cols])

        return data.drop("Target", axis=1), data["Target"].astype(int)
        
    
    def _load_bank(self):
        """Load and preprocess the Bank marketing dataset."""

        file_path = os.path.join(self.data_dir, "bank.csv")
        data = pd.read_csv(file_path, sep=";")
        data.drop(["duration"], axis=1, inplace=True)
        
        cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
        num_cols = ["age", "balance", "day", "campaign", "pdays", "previous"]

        for col in cat_cols:
            data[col] = self._encode_labels(data[col])

        data[num_cols] = self._preprocess_data(data[num_cols])

        data.drop_duplicates(inplace=True)

        return data.drop("y", axis=1), data["y"].astype(int)

    def _load_thyroid(self):
        """Load and preprocess the Thyroid dataset."""

        file_path = os.path.join(self.data_dir, "thyroid.csv")
        data = pd.read_csv(file_path)
        
        for col in data.columns:
            data[col] = data[col].astype("category").cat.codes

        data.drop_duplicates(inplace=True)

        return data.drop("Recurred", axis=1), data["Recurred"]


    def _load_parkinson(self):
        """Load and preprocess the Parkinson dataset."""

        file_path = os.path.join(self.data_dir, "parkinson.data")
        data = pd.read_csv(file_path)

        data = data.drop(["subject#", "motor_UPDRS"], axis=1)
        
        X = self._preprocess_data(data.drop("total_UPDRS", axis=1))
        Y = data["total_UPDRS"]

        return X, Y

    def _load_thoraric(self):

        file_path = os.path.join(self.data_dir, "thoraric.arff")
        raw_data, _ = arff.loadarff(file_path)
        data = pd.DataFrame(raw_data)

        data = data.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        
        bool_cols = data.columns[data.isin(['T', 'F']).any()].tolist()
        for col in bool_cols:
            data[col] = data[col].map({'T': 1, 'F': 0})

        cat_cols = ["DGN", "PRE6", "PRE14"]
        num_cols = ["PRE4", "PRE5", "AGE"]
        for col in cat_cols:
            data[col] = self._encode_labels(data[col])

        data[num_cols] = self._preprocess_data(data[num_cols])

        return data.drop("Risk1Yr", axis=1), data["Risk1Yr"].astype(int)
    
    def _load_compas(self):
        """Load and preprocess the COMPAS recidivism dataset."""

        file_path = os.path.join(self.data_dir, "compas-scores-two-years.csv")
        data = pd.read_csv(file_path)

        data = data[
                (data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'].isin(['F', 'M'])) &
                (data['score_text'].isin(['Low', 'Medium', 'High']))
            ].copy()
        
        data["c_jail_in"] = pd.to_datetime(data["c_jail_in"])
        data["c_jail_out"] = pd.to_datetime(data["c_jail_out"])
        data["length_of_stay"] = (data["c_jail_out"] - data["c_jail_in"]).dt.days
        data.drop(["c_jail_in", "c_jail_out"], axis=1, inplace=True)

        selected_cols = [
            'sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'days_b_screening_arrest', 'decile_score', 'two_year_recid', 'length_of_stay', 'score_text', "age_cat"        
        ]
        data = data[selected_cols]

        cat_cols = ["sex", "age_cat", "race", "c_charge_degree", "score_text"]
        num_cols = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "days_b_screening_arrest", "length_of_stay"]

        for col in cat_cols:
            data[col] = self._encode_labels(data[col])
        data[num_cols] = self._preprocess_data(data[num_cols])

        return data.drop("two_year_recid", axis=1), data["two_year_recid"]

    
    def _load_bike(self):
        """Load and preprocess the Bike Sharing dataset."""

        file_path = os.path.join(self.data_dir, "bike.csv")
        data = pd.read_csv(file_path)
        
        drop_cols = ["instant", "dteday", "casual", "registered"]
        data = data.drop(columns=drop_cols, axis=1)

        num_cols = ["temp", "atemp", "hum", "windspeed"]
        data[num_cols] = self._preprocess_data(data[num_cols])

        return data.drop("cnt", axis=1), data["cnt"]

    def _load_image(self):
        """Load a sample of ImageNet images from local storage."""

        images_path = os.path.join(self.data_dir, "imagenet50")
        X = np.load(os.path.join(images_path, "imagenet50_224x224.npy"))
        Y = np.loadtxt(os.path.join(images_path, "imagenet50_labels.csv"))

        with open(os.path.join(images_path, "imagenet_class_index.json")) as f:
            class_names = [v[1] for v in json.load(f).values()]

        return X, Y, class_names
    
    # --- Model Loading ---
    def load_model(self):

        available_models = ["SVM", "MLP", "XGB", "ResNet50"]
        regression_data = ["parkinson", "bike"]

        if self.model_name not in available_models:
            raise ValueError(f"Model {self.model_name} not found.")

        if self.model_name == "ResNet50":
            if not self.image_data_loaded:
                raise RuntimeError("Load image data before using ResNet50.")
            model = ResNet50(weights='imagenet')
            return model.predict, self.image_class_names
        
        X_train, X_test, Y_train, Y_test = self.load_data()

        if X_train is None:
            raise ValueError("Dataset loading failed.")

        model_path = f"Classifier/{self.dataset_name}/{self.dataset_name}_{self.model_name}.pkl"

        try:
            model = joblib.load(model_path)

            return model.predict if self.dataset_name in regression_data else model.predict_proba
        
        except FileNotFoundError:
            print(f"Model file '{model_path}' not found.")
            return None


