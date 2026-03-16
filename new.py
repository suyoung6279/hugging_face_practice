import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="대신 철벽쳐드립니다", page_icon="🧱")

@st.cache_resource
def load_models():
    zero_shot = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    sim_model = SentenceTransformer("dragonkue/BGE-m3-ko")
    gen_model = pipeline("text-generation", model="Bllossom/llama-3.2-Korean-Bllossom-3B") 
    
    return zero_shot, sim_model, gen_model

with st.spinner("🏗️벽을 쌓고 있는 중이에요.. (초기 1회만 조금 오래 걸려요) ⏳"):
    zeroshot_classifier, model, ko_generator = load_models()

# --- 2. 분석 함수 정의 ---
def zero_shot_classification(text):
    candidate_labels = ["업무 요청", "업무 감사", "일상 안부", "사적 관심"]
    zero_shot = zeroshot_classifier(text, candidate_labels, hypothesis_template="이 문장은 {}에 해당한다.")

    if zero_shot['labels'][0] == '사적 관심':
        st.warning("🚨 1차 경고: 아 이거 좀 수상한데요, 정밀 분석으로 넘어갈게요.")
        return zero_shot['sequence']
    else:
        st.success("✅ 일상 대화인데, 그정도는 아니에요.")
        return None

def second_sentence_check(text):
    target_data = [
        "오빠 오늘 시간 어때요? ㅎㅎ", "주말에 시간 나면 같이 밥 먹을래요?", 
        "이번 주말에 영화 보러 갈래요? 제가 예매할게요!", "언제 시간 내서 단둘이 술 한잔해요.",
        "자기야 너무 고마워 진짜 감동이야 ㅜㅜ", "오늘따라 왜 이렇게 예쁘게 하고 왔어요?", 
        "우리 애기 밥은 먹고 일해용? ㅠㅠ", "너랑 같이 있으니까 시간이 너무 빨리 가는 거 같아.",
        "자요? 갑자기 목소리 듣고 싶어서요.", "어제 누구랑 술 마셨어요? 질투 나려 그러네 ㅋㅋㅋ", 
        "이상형이 어떻게 되세요? 저 같은 스타일은 어때요?", "오늘 하루 종일 네 생각 났어."
    ]
    
    nontarget_data = [
        "네 확인해주셔서 감사합니다.", "업무 협조에 감사드립니다.", "수고하셨습니다. 조심히 들어가세요.",
        "오늘 하루도 즐겁게 보내세요.", "날씨가 많이 추워졌네요. 감기 조심하세요.", 
        "식사는 맛있게 하셨나요?", "주말 편안하게 잘 쉬시길 바랍니다.",
        "도움 주셔서 정말 감사합니다.", "네, 알겠습니다. 신경 써주셔서 고맙습니다.", 
        "좋은 정보 감사합니다. 잘 참고하겠습니다."
    ]
    
    text_vec = model.encode([text])
    target_vecs = model.encode(target_data)
    nontarget_vecs = model.encode(nontarget_data)

    target_score = np.max(cosine_similarity(text_vec, target_vecs))
    nontarget_score = np.max(cosine_similarity(text_vec, nontarget_vecs))
    
    st.info(f"💌 플러팅 일치율: {target_score:.4f} vs 👨‍🦱 일상 대화 일치율: {nontarget_score-0.3:.4f}")
    
    if target_score > nontarget_score - 0.3: 
        st.error("💥 최종 판결: 플러팅이 감지되었습니다! 철벽 문구를 생성합니다.")
        return text
    else:
        st.success("✅ 사적 대화 뉘앙스가 있지만, 선을 넘지 않았습니다. 안심하세요.")
        return None

# --- 3. 웹 화면 UI 구성 ---
st.title("철벽 대신 쳐드립니다. 🧱")
st.markdown("애매한 카톡, 플러팅인지 헷갈리시나요? 저희가 분석해 드릴게요!")
st.divider()

user_input = st.text_area("📩 상대방이 보낸 메시지를 입력하세요:", placeholder="예: 주말에 시간 어때요? 영화나 볼까요 ㅎㅎ")

if st.button("방어 시스템 가동!", type="primary"):
    if not user_input.strip():
        st.warning("메시지를 먼저 입력해주세요!")
    else:
        # 1차 검사
        final_text = zero_shot_classification(user_input)
        
        if final_text is not None:
            # 2차 검사
            final_result = second_sentence_check(final_text)
            
            if final_result is not None:
                st.divider()
                # 3차 텍스트 생성
                with st.spinner("🧱 철벽 문구를 만드는 중입니다..."):
                    prompt =  f"상황: 애인이 있는 상태에서 다른 사람의 호감을 거절함.\n상대방: {final_result}\n단호한 단답형 거절:"
                    
                    generated_output = ko_generator(
                        prompt,
                        max_new_tokens=30,
                        pad_token_id=ko_generator.tokenizer.eos_token_id,       
                        truncation=True,    
                        num_return_sequences=1, 
                        repetition_penalty=1.5, 
                        temperature=0.1 
                    )
                    
                    full_text = generated_output[0]['generated_text']
                    
                    # 에러 방지를 위한 안전한 문자열 자르기
                    try:
                        final_sentence = full_text.split("단호한 단답형 거절:")[1].split('\n')[0].strip()
                    except IndexError:
                        final_sentence = full_text.replace(prompt, "").strip()
                    
                    st.success(f"### 🔥 [추천 철벽 멘트]\n\n> **{final_sentence}**")