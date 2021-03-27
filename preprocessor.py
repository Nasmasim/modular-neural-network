import numpy as np

class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
 
        #find maximum and minimum in data and set a and b according to range [0,1]
        self.maximum = np.amax(data, axis = 0)
        self.minimum = np.amin(data, axis = 0)
        self.a = np.ones(data.shape[1])
        self.b = np.zeros(data.shape[1])



    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """

        #perform min max normalization : Scaling the smallest value to a and largest value to b
        normalized_data = self.a + (data - self.minimum) * (self.b-self.a) / (self.maximum - self.minimum)
        return normalized_data


    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """

        #revert min max normalization : Scaling a to smallest value and b to largest value
        reverted_data = (data - self.a) * (self.maximum - self.minimum) / (self.b-self.a) + self.minimum 
        return reverted_data


