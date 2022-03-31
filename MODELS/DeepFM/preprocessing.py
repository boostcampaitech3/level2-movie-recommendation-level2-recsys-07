import pandas as pd
import os

def main():
    if os.path.isfile('data/train/rating.csv'):
        print('rating.csv is already exist')
        return

    # Rating df 생성
    rating_data = "/opt/ml/input/data/train/train_ratings.csv"
    raw_rating_df = pd.read_csv(rating_data)
    raw_rating_df['rating'] = 1.0 # implicit feedback
    raw_rating_df.drop(['time'],axis=1,inplace=True)
    raw_rating_df.to_csv('data/train/rating.csv')
    print("Create rating df!")


def genre():
    if os.path.isfile('data/train/genre.csv'):
        print('gnere.csv is already exist')
        return

    # Genre df 생성
    genre_data = "/opt/ml/input/data/train/genres.tsv"
    raw_genre_df = pd.read_csv(genre_data, sep='\t')
    raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

    genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
    raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경
    raw_genre_df.to_csv('data/train/genre.csv')
    print("Create genre df!")


if __name__ == "__main__":
    main()
    genre()