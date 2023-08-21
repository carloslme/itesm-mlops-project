from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from preprocess.preprocess_data import (
    MissingIndicator,
    ExtractLetters,
    CategoricalImputer,
    NumericalImputer,
    RareLabelCategoricalEncoder,
    OneHotEncoder,
    FeatureSelector,
    OrderingFeatures
)

class TitanicDataPipeline:
    """
    A class representing the Titanic data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS_WITH_NA (list): A list of categorical variables with missing values.
        NUMERICAL_VARS_WITH_NA (list): A list of numerical variables with missing values.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Titanic data processing pipeline.
    """
    
    def __init__(self, seed_model, numerical_vars, categorical_vars_with_na,
                 numerical_vars_with_na, categorical_vars, selected_features):
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.CATEGORICAL_VARS_WITH_NA = categorical_vars_with_na
        self.NUMERICAL_VARS_WITH_NA = numerical_vars_with_na
        self.CATEGORICAL_VARS = categorical_vars
        self.SEED_MODEL = seed_model
        self.SELECTED_FEATURES = selected_features
        
        
    def create_pipeline(self):
        """
        Create and return the Titanic data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                                ('missing_indicator', MissingIndicator(variables=self.NUMERICAL_VARS)),
                                ('cabin_only_letter', ExtractLetters()),
                                ('categorical_imputer', CategoricalImputer(variables=self.CATEGORICAL_VARS_WITH_NA)),
                                ('median_imputation', NumericalImputer(variables=self.NUMERICAL_VARS_WITH_NA)),
                                ('rare_labels', RareLabelCategoricalEncoder(tol=0.05, variables=self.CATEGORICAL_VARS)),
                                ('dummy_vars', OneHotEncoder(variables=self.CATEGORICAL_VARS)),
                                ('feature_selector', FeatureSelector(self.SELECTED_FEATURES)),
                                ('aligning_feats', OrderingFeatures()),
                                ('scaling', MinMaxScaler()),
                              ]
        )
        return self.PIPELINE
    
    def fit_logistic_regression(self, X_train, y_train):
        """
        Fit a Logistic Regression model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_regression_model (LogisticRegression): The fitted Logistic Regression model.
        """
        logistic_regression = LogisticRegression(C=0.0005, class_weight='balanced', random_state=self.SEED_MODEL)
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)
        logistic_regression.fit(pipeline.transform(X_train), y_train)
        return logistic_regression
    
    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        pipeline = self.create_pipeline()
        return pipeline.transform(X_test)