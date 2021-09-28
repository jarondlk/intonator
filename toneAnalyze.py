import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def toneAnalyze(sentence):
    model = pickle.load(open('datas/finalized_model.sav', 'rb'))

    lemtext_csv = pd.read_csv("datas/lemtext_new.csv")
    lemtext = lemtext_csv.iloc[10:, 1]  # Dataframe -> series

    count_vectorizer = CountVectorizer(max_features=1000, min_df=8)
    count_vectorizer.fit(lemtext.astype('U'))

    inp = np.array(sentence).reshape((1, -1))

    df_temp = count_vectorizer.transform(inp.ravel())

    if model.predict(df_temp.toarray()) == [0.]:
        return "negative"
    else:
        return "positive"
