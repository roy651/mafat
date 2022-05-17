from model import model
import pandas as pd

df = pd.read_csv('test1_labeled.csv')
df.drop('Num_People', axis=1, inplace=True)

test_model = model()
test_model.load('./')
res = test_model.predict(df)
print(res)