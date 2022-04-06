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


def director():
    if os.path.isfile('data/train/director.csv'):
        print('director.csv is already exist')
        return

    # Director df 생성
    director_data = "/opt/ml/input/data/train/directors.tsv"
    raw_director_df = pd.read_csv(director_data, sep='\t')
    raw_director_df = raw_director_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

    director_dict = {director:i for i, director in enumerate(set(raw_director_df['director']))}
    raw_director_df['director']  = raw_director_df['director'].map(lambda x : director_dict[x]) #genre id로 변경
    raw_director_df.to_csv('data/train/director.csv')
    print("Create director df!")


def writer():
    if os.path.isfile('data/train/writer.csv'):
        print('writer.csv is already exist')
        return

    # Writer df 생성
    writer_data = "/opt/ml/input/data/train/writers.tsv"
    raw_writer_df = pd.read_csv(writer_data, sep='\t')
    raw_writer_df = raw_writer_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

    writer_dict = {writer:i for i, writer in enumerate(set(raw_writer_df['writer']))}
    raw_writer_df['writer']  = raw_writer_df['writer'].map(lambda x : writer_dict[x]) #genre id로 변경
    raw_writer_df.to_csv('data/train/writer.csv')
    print("Create writer df!")

def join_genre_writer():
    
    #-- 경로 설정
    GENRE_PATH = './data/train/genre.csv'
    WRITER_PATH = './data/train/writer.csv'
    OUTPUT_PATH = './data/train/genre_writer.csv'
    
    #-- 이미 파일이 만들어진 적 있는 경우 pass
    if os.path.isfile(OUTPUT_PATH):
        print(f'{OUTPUT_PATH} already exists.')
        return
    
    #-- read csv (genre, writer) 
    genre_df = pd.read_csv(GENRE_PATH)
    writer_df = pd.read_csv(WRITER_PATH)
    
    #-- inner join 2 dataframes
    joined_df = pd.merge(left=genre_df, right=writer_df, how='inner', on='item')
    
    #-- save joined dataframe as "genre_writer.csv"
    joined_df.to_csv(OUTPUT_PATH, columns=['item', 'genre', 'writer'], index=False)


if __name__ == "__main__":
    main()
    genre()
    director()
    writer()
    join_genre_writer()