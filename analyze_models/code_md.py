results = pd.read_csv('{}_cvresults.csv')
pipeline = load('{}_bestestimator.joblib')
results.head()
