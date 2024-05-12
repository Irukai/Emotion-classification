import pandas as pd

def preprocess(test_file, select_file):
    # test_file = './data/test.csv'
    # select_file = './data/Test_Dataset.csv'
    # CSV 파일을 DataFrame으로 읽기
    df = pd.read_csv(test_file, encoding='cp949')

    # 각 상황별로 5개씩 샘플링
    frames = []
    for situation in ['sad', 'anger', 'fear', 'disgust', 'neutral']:
        sampled_situation = df[df['상황'] == situation].sample(n=5, random_state=1)
        frames.append(sampled_situation)

    # 데이터 프레임들을 합치기
    sampled_df = pd.concat(frames)

    # 원하는 컬럼만 선택
    result_df = sampled_df[['wav_id', '발화문', '상황']]

    # 결과 확인
    print(result_df)

    # result_df를 CSV 파일로 저장
    result_df.to_csv(select_file, index=False, encoding='utf-8-sig')
