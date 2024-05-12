### 모델 저장 -> STT_Emotion_Classifier.py에서 모델 불러서 test진행하기



import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from koBERT import predict # koBERT.py에서 predict() 함수를 불러오자.
from preprocessing import preprocess

# 모델 정의 (동일한 모델 구조가 필요)
model, vocab = get_pytorch_kobert_model()

# 모델 상태 로드
model.load_state_dict(torch.load('kobert_model.pth'))



from STT import speech_to_text

# set the audio file
files_path = "./data/test/"
test_path = "./data/select_data.csv"

# 테스트 데이터 전처리
preprocess(files_path, test_path) # test_path로 test에서 각 상황별 5개씩 뽑은 데이터가 저장됨.


test_data = pd.read_csv(test_path)
wav_id = test_data.wav_id + '.wav'

stt_text = speech_to_text(wav_id, files_path)
predict_sentences = [[s, '0'] for s in stt_text]

labels = predict(predict_sentences) # labels = [0, 1, 2, 3, 2, 1, 4, 0, ... 2] 

# 레이블에 대한 감정 매핑
emotion = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'surprised', 4: 'neutral'}

result = pd.DataFrame({
   'text': stt_text,
   'labels': labels 
})

# labels 컬럼의 숫자를 단어로 매핑
result['labels'] = result['labels'].map(emotion)

# 결과 출력
print(result)





## 아래 코드는 모델 저장하는 코드를 넣고, predict 함수를 수정해야 한다는 의미이다.

'''
# 모델 학습 후, 모델의 상태를 파일로 저장
torch.save(model.state_dict(), 'kobert_model.pth')



def predict(predict_sentences): # input = 감정분류하고자 하는 sentence

    another_test = BERTDataset(predict_sentences, 0, 1, tok, vocab, max_len, True, False) # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        ## 여기는 out이 어떻게 나오는지 확인해봐야 할 것 같음.

        test_eval = []
        for i in out: # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

'''