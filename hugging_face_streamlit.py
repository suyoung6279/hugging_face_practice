import streamlit as st
from transformers import pipeline
from konlpy.tag import Okt

@st.cache_resource
def load_models():
    clf = pipeline('zero-shot-classification', model='MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli')
    okt_model = Okt()
    return clf, okt_model

st.write("⏳ 모델을 불러오는 중입니다. 잠시만 기다려주세요...")
ko_zero_shot_clf, okt = load_models()
st.success("모델 로드 완료!")

# 2. 웹 UI 구성
st.title("☁️ 사회초년생의 언어 필터기")
st.markdown("작성한 문장을 분석하여 무례하거나 공격적인 표현을 걸러냅니다.")

# 텍스트 입력창
input_text = st.text_area("분석할 문장들을 입력하세요 (엔터로 줄바꿈):", height=150)

if st.button("필터링 시작"):
    if input_text:
        text_list = [t.strip() for t in input_text.split('\n') if t.strip()]
        
        st.markdown("---")
        st.subheader("🔍 1단계: 문장 어조 분석")
        
        sentence_labels = ["무례하고 공격적임", "예의 바르고 정중함"]
        shot_word = ko_zero_shot_clf(text_list, sentence_labels, hypothesis_template="이 문장은 직장 상사에게 {} 느낌을 준다.")
        
        if not isinstance(shot_word, list):
            shot_word = [shot_word]

        word_list = []
        target_sentences = [] 

        for k in range(len(shot_word)):
            top_sentence_label = shot_word[k]['labels'][0]
            current_sentence = shot_word[k]['sequence']
            
            if top_sentence_label == '무례하고 공격적임':
                st.error(f"🚨 주의 필요: '{current_sentence}'")
                target_sentences.append(current_sentence)

                result = okt.pos(current_sentence)
                i = 0
                while i < len(result):
                    word, pos = result[i]

                    if pos == 'Modifier':
                        if i + 1 < len(result): 
                            word1, pos1 = result[i+1]
                            word = word + word1 
                            i += 2
                            word_list.append(word)
                        else:
                            i += 1
                    else:
                        i += 1

                    if pos in ['Noun', 'Verb', 'Adjective', 'Exclamation']:
                        word_list.append(word)
            else:
                st.success(f"✅ 통과: '{current_sentence}'")

        word_list = list(set(word_list))

        if word_list:
            st.markdown("---")
            st.subheader("🕵️‍♂️ 2단계: 세부 단어 뉘앙스 검사")
            
            detailed_labels = ["공격적인 비속어", "무례한 반말", "짜증과 불평", "일반적인 단어"]
            villain_results = []
            
            for word in word_list:
                context_sentence = " ".join(target_sentences)
                
                res = ko_zero_shot_clf(word, detailed_labels, 
                                       hypothesis_template=f"'{context_sentence}'라는 문장에서 '{word}'는 {{}} 느낌을 준다.")
                
                top_label = res['labels'][0]
                
                if top_label in ["공격적인 비속어", "무례한 반말", "짜증과 불평"]:
                    villain_results.append({"word": word, "label": top_label})

            if villain_results:
                st.write("### ⚠️ 수정 제안 단어 목록")
                for v in villain_results:
                    st.warning(f"**{v['word']}** (판정: {v['label']}) -> 수정이 필요합니다.")
            else:
                st.info("무례한 문장이었지만, 특정할 만한 나쁜 단어는 찾지 못했습니다.")
                
    else:
        st.warning("텍스트를 입력해 주세요!")