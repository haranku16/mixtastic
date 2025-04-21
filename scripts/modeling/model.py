from typing import List
from scripts.types import Stems, Mix

class Model:
    '''
    Base class for all models.
    '''
    def __init__(self):
        pass

    def fit(self, stems: List[Stems]):
        '''
        Fit the model to the data.

        Args:
            X: List[Stems] - The input stems.
        '''
        raise NotImplementedError("Subclasses must implement this method.") 

    def predict(self, X: List[Stems]) -> List[Mix]:
        '''
        Predict the target variable.

        Args:
            X: List[Stems] - The input stems.

        Returns:
            List[Mix] - The predicted target mix.
        '''
        raise NotImplementedError("Subclasses must implement this method.")
