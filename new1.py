import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="철벽 대신 쳐드립니다", page_icon="🧱", layout="centered")

# ── 커스텀 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&family=Black+Han+Sans&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #EEF4FF 0%, #F5F0FF 40%, #FFF0F5 100%);
    min-height: 100vh;
}

.block-container {
    max-width: 680px !important;
    padding: 0 1.8rem 5rem !important;
}

/* ── 히어로 ── */
.hero-wrap {
    text-align: center;
    padding: 3.6rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,91,255,0.10);
    color: #635BFF;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Black Han Sans', sans-serif;
    font-size: 2.8rem;
    color: #111;
    line-height: 1.25;
    margin: 0 0 0.6rem;
    letter-spacing: -1px;
}
.hero-title .accent { color: #635BFF; }
.hero-sub {
    font-size: 1rem;
    color: #888;
    margin: 0;
    line-height: 1.7;
}

/* ── 구분선 ── */
hr {
    border: none !important;
    border-top: 1px solid #E8E8EE !important;
    margin: 1.6rem 0 !important;
}

/* ── 입력 레이블 ── */
.input-label {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #999;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── textarea ── */
textarea {
    background: #FFFFFF !important;
    border: 1.5px solid #E5E5F0 !important;
    border-radius: 16px !important;
    color: #111 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.1rem !important;
    resize: none !important;
    box-shadow: 0 2px 10px rgba(80,70,200,0.05) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus {
    border-color: #635BFF !important;
    box-shadow: 0 0 0 4px rgba(99,91,255,0.10) !important;
}
textarea::placeholder { color: #BBBBC8 !important; }

/* ── 버튼 ── */
.stButton > button {
    width: 100%;
    background: #635BFF !important;
    color: #fff !important;
    font-family: 'Black Han Sans', sans-serif !important;
    font-size: 1.08rem !important;
    letter-spacing: 0.8px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,91,255,0.30) !important;
}
.stButton > button:hover {
    background: #4F46E5 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,91,255,0.38) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── 알림 박스 ── */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
    font-size: 0.93rem !important;
}

/* ── 단계 칩 ── */
.step-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,91,255,0.08);
    color: #635BFF;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin: 1.1rem 0 0.5rem;
}

/* ── 판결 배너 ── */
.verdict-banner {
    border-radius: 16px;
    padding: 1rem 1.3rem;
    margin: 0.4rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1.5;
}
.verdict-danger {
    background: #FFF2F5;
    color: #D63060;
    border: 1.5px solid #FFCCD8;
}
.verdict-safe {
    background: #F0FFF6;
    color: #1A9E5C;
    border: 1.5px solid #C0EDD4;
}
.verdict-icon { font-size: 1.3rem; flex-shrink: 0; }

/* ── 점수 카드 ── */
.score-card {
    background: #FFFFFF;
    border: 1.5px solid #EEEEF8;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0 0.9rem;
    box-shadow: 0 2px 12px rgba(80,70,200,0.05);
}
.score-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #AAA;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.score-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.55rem 0;
}
.score-name {
    font-size: 0.82rem;
    color: #555;
    width: 120px;
    flex-shrink: 0;
    font-weight: 500;
}
.score-bar-bg {
    flex: 1;
    height: 7px;
    background: #F0F0F7;
    border-radius: 99px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.score-val {
    font-size: 0.82rem;
    font-weight: 700;
    width: 44px;
    text-align: right;
}

/* ── 결과 카드 ── */
.result-card {
    background: linear-gradient(135deg, #635BFF 0%, #9B5CF6 100%);
    border-radius: 22px;
    padding: 1.8rem 2rem;
    margin-top: 0.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 36px rgba(99,91,255,0.30);
}
.result-card::after {
    content: '🧱';
    position: absolute;
    right: 1.4rem;
    bottom: -0.8rem;
    font-size: 5.5rem;
    opacity: 0.10;
}
.result-label {
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.60);
    margin-bottom: 0.7rem;
}
.result-text {
    font-family: 'Black Han Sans', sans-serif;
    font-size: 1.5rem;
    color: #FFFFFF;
    line-height: 1.6;
    word-break: keep-all;
    position: relative;
    z-index: 1;
}

/* ── spinner ── */
.stSpinner > div { color: #635BFF !important; }
</style>
""", unsafe_allow_html=True)

# ── 히어로 ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">철벽, <span class="accent">대신</span><br>쳐드립니다</p>
  <p class="hero-sub">애매한 카톡이 플러팅인지 헷갈리시나요?<br>AI가 분석하고 철벽 멘트까지 만들어 드려요.</p>
</div>
""", unsafe_allow_html=True)

# ── 모델 로딩 ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    zero_shot = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    sim_model = SentenceTransformer("dragonkue/BGE-m3-ko")
    gen_model = pipeline("text-generation", model="Bllossom/llama-3.2-Korean-Bllossom-3B")
    return zero_shot, sim_model, gen_model

with st.spinner("🏗️ 벽을 쌓고 있는 중이에요.. (초기 1회만 조금 오래 걸려요) ⏳"):
    zeroshot_classifier, model, ko_generator = load_models()

# ── 분석 함수 ──────────────────────────────────────────────────
def zero_shot_classification(text):
    candidate_labels = ["일상 안부", "사적 관심"]
    zero_shot = zeroshot_classifier(text, candidate_labels, hypothesis_template="이 문장은 {}에 해당한다.")
    if zero_shot['labels'][0] == '사적 관심':
        st.markdown("""
        <div class="verdict-banner verdict-danger">
          <span class="verdict-icon">🚨</span>
          <span>1차 경고 — 이거 수상한데요, 정밀 분석으로 넘어갑니다.</span>
        </div>""", unsafe_allow_html=True)
        return zero_shot['sequence']
    else:
        st.markdown("""
        <div class="verdict-banner verdict-safe">
          <span class="verdict-icon">✅</span>
          <span>1차 통과 — 이정도는 괜찮아요 </span>
        </div>""", unsafe_allow_html=True)
        return None

def second_sentence_check(text):
    target_data = [
        "오빠 오늘 시간 어때요? ㅎㅎ", "주말에 시간 나면 같이 밥 먹을래요?",
        "이번 주말에 영화 보러 갈래요? 제가 예매할게요!", "언제 시간 내서 단둘이 술 한잔해요.",
        "자기야 너무 고마워 진짜 감동이야 ㅜㅜ", "오늘따라 왜 이렇게 예쁘게 하고 왔어요?",
        "나랑 사귈래?", "오늘따라 왜 이렇게 예쁘게 하고 왔어요?",
        "핸드크림 바를래?", "너랑 같이 있으니까 시간이 너무 빨리 가는 거 같아.",
        "자? 갑자기 목소리 듣고 싶어서.", "어제 누구랑 술 마셨어? 질투 나려 그러네 ㅋㅋㅋ",
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
    target_score = float(np.max(cosine_similarity(text_vec, model.encode(target_data))))
    nontarget_score = float(np.max(cosine_similarity(text_vec, model.encode(nontarget_data))))

    t_pct = int(target_score * 100)
    n_pct = int(max(nontarget_score, 0) * 100)

    st.markdown(f"""
    <div class="score-card">
      <div class="score-card-title">📊 유사도 분석 결과</div>
      <div class="score-row">
        <span class="score-name">💌 플러팅 일치율</span>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:{t_pct}%;background:linear-gradient(90deg,#FF5A7E,#FF8C42);"></div>
        </div>
        <span class="score-val" style="color:#FF5A7E;">{target_score:.3f}</span>
      </div>
      <div class="score-row">
        <span class="score-name">💬 일상 대화 일치율</span>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:{n_pct}%;background:linear-gradient(90deg,#635BFF,#9B5CF6);"></div>
        </div>
        <span class="score-val" style="color:#635BFF;">{max(nontarget_score, 0):.3f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if target_score > nontarget_score - 0.3:
        st.markdown("""
        <div class="verdict-banner verdict-danger">
          <span class="verdict-icon">🚧</span>
          <span>철벽 공사에 돌입합니다!</span>
        </div>""", unsafe_allow_html=True)
        return text
    else:
        st.markdown("""
        <div class="verdict-banner verdict-safe">
          <span class="verdict-icon">✅</span>
          <span>최종 판결 — 사적인 관심은 있는 것 같은데 좀 더 지켜봐요.</span>
        </div>""", unsafe_allow_html=True)
        return None

# ── UI 입력 ────────────────────────────────────────────────────
st.markdown('<p class="input-label">📩 상대방 메시지</p>', unsafe_allow_html=True)

user_input = st.text_area(
    label="상대방 메시지",
    placeholder="예: 주말에 시간 어때요? 영화나 볼까요 ㅎㅎ",
    height=120,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🧱 철벽 공사 시작!", type="primary"):
    if not user_input.strip():
        st.warning("메시지를 먼저 입력해주세요!")
    else:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="step-chip">🔎 STEP 1 &nbsp;·&nbsp; 의도 분류</div>', unsafe_allow_html=True)

        final_text = zero_shot_classification(user_input)

        if final_text is not None:
            st.markdown('<div class="step-chip">🔬 STEP 2 &nbsp;·&nbsp; 유사도 정밀 분석</div>', unsafe_allow_html=True)
            final_result = second_sentence_check(final_text)

            if final_result is not None:
                with st.spinner("🧱 철벽 문구를 조각하는 중..."):
                    prompt = f"상황: 애인이 있는 상태에서 다른 사람의 호감을 거절함.\n상대방: {final_result}\n단호한 단답형 거절:"
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
                    try:
                        final_sentence = full_text.split("단호한 단답형 거절:")[1].split('\n')[0].strip()
                    except IndexError:
                        final_sentence = full_text.replace(prompt, "").strip()

                st.markdown('<div class="step-chip">🔥 STEP 3 &nbsp;·&nbsp; 철벽 멘트 완성</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-card">
                  <p class="result-label">🔥 추천 철벽 멘트</p>
                  <p class="result-text">"{final_sentence}"</p>
                </div>
                """, unsafe_allow_html=True)