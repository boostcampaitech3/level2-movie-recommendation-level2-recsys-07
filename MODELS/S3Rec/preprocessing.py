import pandas as pd
import json

def main():
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array

    writers_df = pd.read_csv("../data/train/writers.tsv", sep="\t")
    array, index = pd.factorize(writers_df["writer"])
    writers_df["writer"] = array

    genres_df_list = genres_df.groupby("item")["genre"].apply(list)
    writers_df_list = writers_df.groupby("item")["writer"].apply(list)
    
    result = dict()
    for item in writers_df_list.index:
        result[item] = [genres_df_list[item], writers_df_list[item]]
        
    with open('./data/train/genre_writer.json','w') as f:
        json.dump(result, f)
if __name__ == "__main__":
    main()
