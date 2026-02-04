import pandas as pd
from collections import Counter

df = pd.read_csv("data/data.csv")
df2 = pd.read_csv("data/data2.csv")
df3 = pd.read_csv("data/data3.csv")

df = pd.concat([df, df2, df3], ignore_index=True)

# Same sentiment logic
def label_sentiment(rating):
    if rating >= 4:
        return 1
    else:
        return 0

df['sentiment'] = df['Ratings'].apply(label_sentiment)

negative_reviews = df[df['sentiment'] == 0]

all_words = " ".join(negative_reviews['Review text'].astype(str)).split()

common_words = Counter(all_words).most_common(20)

print("Top Pain Point Words:")
print(common_words)
