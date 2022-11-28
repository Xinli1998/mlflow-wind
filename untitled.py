import pandas as pd
import mlflow
import numpy as np
import sys
import click

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

# mlflow.set_tracking_uri("http://localhost:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("xinl")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt 
from urllib.parse import urlparse
from sklearn.neural_network import MLPRegressor

# alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
# # l1_ratio = 0.5
# max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
# print(sys.argv,len(sys.argv))
# degree = int(sys.argv[3]) if len(sys.argv) > 3 else 4
# number_of_splits = int(sys.argv[4]) if len(sys.argv) > 4 else 5

@click.command()
@click.option("--alpha", default=0.1, type=float)
@click.option("--max_iter", default=10000, type=int)
@click.option("--degree", default=4, type=int)
@click.option("--number_of_splits", default=5, type=int)

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
def workflow(alpha, max_iter, degree, number_of_splits):

    with mlflow.start_run(run_name='mlflow-wind'):
        # TODO: Insert path to dataset
        df = pd.read_json("dataset.json", orient="split")

        df = df.dropna()

        splitter = ColumnTransformer([
        ('Standard', StandardScaler(), ['Speed']),

        ('Onehot',OneHotEncoder(handle_unknown='ignore'), ['Direction'])],
        remainder= 'passthrough')

        #     pipeline= Pipeline([
        #         # Here you can add your preproccesing transformers
        #         # And you can add your model as the final step
        #           ('transformer', splitter),
        #             ('poly_features', PolynomialFeatures (degree=degree, include_bias=False)),
        #         ('elastic_model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)),  

        #     ])

        pipeline = Pipeline([
            # Here you can add your preproccesing transformers
            # And you can add your model as the final step
              ('transformer', splitter ),
                    ('poly_features', PolynomialFeatures (degree=degree, include_bias=False)),
            ('rnn_model', MLPRegressor(max_iter = max_iter,random_state=1, alpha=alpha )),  

        ])

        # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
        metrics = [
            ("MAE", mean_absolute_error, []),
            ("MSE" , mean_squared_error, []),
            ('R2',r2_score,[])
        ]

        X = df[["Speed","Direction"]]
        y = df["Total"]



        #TODO: Log your parameters. What parameters are important to log?
        #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
        print(alpha, max_iter, degree, number_of_splits)
    #     mlflow.log_param("alpha", alpha)
        #     mlflow.log_param("l1_ratio", l1_ratio)
    #     mlflow.log_param("max_iter" , max_iter)
    #     mlflow.log_param("degree", degree)
    #     mlflow.log_param("number_of_splits", number_of_splits)

        acc = []
        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
            pipeline.fit(X.iloc[train],y.iloc[train])
            predictions = pipeline.predict(X.iloc[test])
            truth = y.iloc[test]
            acc.append(pipeline.score(X.iloc[train],y.iloc[train]))


            plt.figure(figsize=(8, 4.5))
            plt.plot(truth.index, truth.values, label="Truth")
            plt.plot(truth.index, predictions, label="Predictions")
            plt.xticks(fontproperties = 'Times New Roman', size = 10,rotation=50)
            plt.show()

            # Calculate and save the metrics for this fold
            for name, func, scores in metrics:

                score = func(truth, predictions)
                scores.append(score)

        # Log a summary of the metrics
        for name, _, scores in metrics:
                # NOTE: Here we just log the mean of the scores. 
                # Are there other summarizations that could be interesting?
                mean_score = sum(scores)/number_of_splits
                print(mean_score)
                mlflow.log_metric(f"mean_{name}", mean_score)
        mlflow.log_metric(f"mean_accuracy",np.mean(acc))

    #         mlflow.sklearn.log_model(pipeline, "models")
        mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model"
    )
#     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#     # Model registry does not work with file store
#     if tracking_url_type_store != "file":

#         # Register the model
#         # There are other ways to use the Model Registry, which depends on the use case,
#         # please refer to the doc for more information:
#         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#         mlflow.sklearn.log_model(pipeline, "model", registered_model_name="ElasticnetWineModel")
#     else:
#         mlflow.sklearn.log_model(pipeline, "model")

if __name__ == '__main__':
    workflow()
