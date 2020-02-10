import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.util.Preprocessor import *


class TestPreprocessor:

    def test_normalize(self):
        preprocessor = Preprocessor()
        df = pd.DataFrame([[1, 3], [2, 4]], columns=['a', 'b'])
        normalized = preprocessor.normalize(df, ['a', 'b'])
