import mlflow
print("Printing tracking uri below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("https://127.0.0.1:5000")
print("Printing new tracking uri below")
print(mlflow.get_tracking_uri())
print("\n")