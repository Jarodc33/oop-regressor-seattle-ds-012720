import numpy as np

class MeanRegressor():
    def fit(self, X, y):
        '''
        `X` is a two-dimensional matrix (nested NumPy array, nested Python list, or Pandas dataframe) of data rows and features.  Your model will be ignoring it.
        - `y` is a list (NumPy array or Python list) representing the target variable
        - The model should determine the mean of `y` and store it, to be used in the `predict` method
        - This method does not return anything
        '''
        if len(X) != len(y):
            raise Exception('The number of rows in X must be the same as the length of Y in order to fit the data')
        self.mean = y.mean()

    def predict(self, X):
        '''
        - `X` is a two-dimensional matrix.  Your model will be ignoring its features, and only using the count of rows.
        - This method returns the mean of the training data for each row of `X`, i.e. a list containing the same number repeated as many times as necessary.
        '''
        return np.repeat(self.mean, len(X))
            
    def score(self, X, y):
        '''
        - `X` is a two-dimensional matrix and `y` is a list of target variables
        - This method will compute the R<sup>2</sup> for how well the features of `X` are able to predict the target of `y`.
          As a reminder, R<sup>2</sup> is calculated as 1 - `residual sum of squares`/`total sum of squares`, where
          `residual sum of squares` is the sum of all ((y_true - y_pred)<sup>2</sup>) and `total sum of squares` is the sum of all ((y_true - y_pred.mean())<sup>2</sup>).
          So, if you are scoring with the same `y` that was used for the `fit`, you should expect the score to be exactly zero.
        '''
        rss = sum((y - self.predict(X)) ** 2)
        tss = sum((y - y.mean()) ** 2)
        return 1 - rss / tss