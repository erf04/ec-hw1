from preparation import *


# part 1 
df = load_data("data.xlsx")
df_clean = clean_data(df)
summary = summarize_data(df_clean)
summary.head()
print(summary)