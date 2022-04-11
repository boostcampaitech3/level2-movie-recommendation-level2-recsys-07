import pandas as pd


def main():
    # genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    # array, index = pd.factorize(genres_df["genre"])
    # genres_df["genre"] = array
    # genres_df.groupby("item")["genre"].apply(list).to_json(
    #     "data/Ml_item2attributes.json"
    # )
    writers_df = pd.read_csv("../data/train/writers.tsv", sep="\t")
    array, index = pd.factorize(writers_df["writer"])
    writers_df["writer"] = array
    writers_df.groupby("item")["writer"].apply(list).to_json(
        "data/Ml_item2attributes.json"
    )

if __name__ == "__main__":
    main()
