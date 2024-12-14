import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../NBA_PBP_2015-19_enhanced.csv")

# plot a bar chart of the "PlayType" column using plt
# the y-axis should be a proportion, labeled "Proportion"
# the x-axis should be labeled "PlayType" and the title should be "Play Types"
# the bars should be sorted by frequency
df["PlayType"].value_counts(normalize=True).plot(kind="bar")
plt.title("Play Types")
plt.ylabel("Proportion")
plt.xlabel("PlayType")

# The x-axis labels are too close together, so rotate them 45 degrees
plt.xticks(rotation=45)

# add labels to the bars
for i, v in enumerate(df["PlayType"].value_counts(normalize=True)):
    plt.text(i, v + 0.003, f"{v:.4f}", ha="center")

plt.show()




# save
plt.savefig("plot.png")
