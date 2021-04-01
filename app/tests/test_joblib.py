from joblib import load, dump

classifier = load('../assets/lr_model.joblib')

print(classifier)