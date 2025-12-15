# K팝 데몬 헌터스 팬덤 분석 대시보드
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
from itertools import combinations
import os

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# 텍스트 처리
from konlpy.tag import Okt
from wordcloud import WordCloud

# 네트워크 분석
import networkx as nx

# 한글 폰트 설정  - Pretendard 폰트 사용
import matplotlib.font_manager as fm

# Pretendard 폰트 경로 설정
font_path = 'font/Pretendard-Regular.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정 (강의록 11.ipynb)
st.set_page_config(
    page_title='K팝 데몬 헌터스 팬덤 분석',
    page_icon='🎵',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get help': 'https://docs.streamlit.io',
        'Report a bug': 'https://streamlit.io',
        'About': '### K팝 데몬 헌터스 팬덤 분석 대시보드 \n - C221088 알렉산더'
    }
)

# 불용어 정의
# 불용어 사전 만들기
stop_str = '예정 에 가 이은 을 를 의 도 또한 더 를 위해 에게 에게서 에게로 부터 어 우선 간 이후 하는 입니다 할 합니다'
# 불용어 문자열을 ' '로 분리한 후 set으로 변환
stop_words = set(stop_str.split(' '))

# 텍스트 정제 함수
def cleanString(text):
    """텍스트 정제 함수"""
    # HTML 태그 제거 (강의록 13.ipynb)
    pattern = r'(<[^>]*>)'
    text = re.sub(pattern=pattern, repl='', string=text)
    
    # 특수문자 제거
    pattern = r'[^\w\s\n]'
    text = re.sub(pattern=pattern, repl='', string=text)
    
    return text

# 캐싱 함수 정의
@st.cache_data
def load_data():
    """데이터 로드 함수"""
    df = pd.read_csv('data/naver_news.csv')
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# 사이드바 구성
# 사이드바 설정
st.sidebar.title('🎵 K팝 데몬 헌터스')
st.sidebar.divider()  # 구분선

# 학번, 이름 표시
st.sidebar.info('**C221088 최유빈**')

st.sidebar.write('### 📊 분석 옵션')

# 위젯 1: 체크박스
show_raw_data = st.sidebar.checkbox('원본 데이터 보기')

# 위젯 2: 슬라이더
top_n_words = st.sidebar.slider('워드클라우드 단어 수', 10, 100, 50)

# 위젯 3: 셀렉트박스
network_min_weight = st.sidebar.selectbox(
    '네트워크 최소 연결 강도',
    [3, 5, 10, 15, 20]
)

# 위젯 4: 라디오 버튼
chart_theme = st.sidebar.radio(
    '차트 색상 테마',
    ['기본', '다크', '컬러풀']
)

# 위젯 5: 멀티셀렉트
analysis_options = st.sidebar.multiselect(
    '분석 항목 선택',
    ['시계열 분석', '키워드 추이 분석', '키워드 빈도 분석', '워드클라우드', '네트워크 분석'],
    default=['시계열 분석', '키워드 추이 분석', '키워드 빈도 분석', '워드클라우드', '네트워크 분석']
)

st.sidebar.divider()  # 구분선
st.sidebar.caption('© 2025 데이터시각화 3차 시험')

# 메인 페이지

# 타이틀
st.title('🎵 K팝 데몬 헌터스 팬덤 분석 대시보드')
st.markdown('**C221088 최유빈** | 데이터시각화 3차 시험')
st.divider()  # 구분선

# 1. 작품 기본 정보 섹션
st.header('📺 작품 기본 정보')

# 컬럼 레이아웃
col1, col2 = st.columns([1, 2])

with col1:
    # 이미지 출력
    if os.path.exists('data/poster.jpg'):
        st.image('data/poster.jpg', caption='K팝 데몬 헌터스 포스터', use_container_width=True)
    elif os.path.exists('data/poster.png'):
        st.image('data/poster.png', caption='K팝 데몬 헌터스 포스터', use_container_width=True)
    else:
        st.image('https://via.placeholder.com/300x400?text=Poster', 
                 caption='K팝 데몬 헌터스', use_container_width=True)

with col2:
    # 작품 정보
    st.subheader('K-Pop Demon Hunters')
    
    # Pandas 데이터프레임 출력
    info_df = pd.DataFrame({
        '항목': ['개봉일', '채널', '감독', '장르'],
        '내용': ['2025년 6월 20일', '넷플릭스', '매기 강, 크리스 아펠한스', '판타지, 액션, 음악']
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.write('#### 📖 줄거리')
    st.write('''
   세계적인 팬덤을 거느린 최정상 K-Pop 걸그룹. 화려한 조명 아래서 완벽한 퍼포먼스를 보여주는 그들이지만, 무대 뒤에는 아무도 모르는 비밀이 있습니다. 바로 사악한 **악귀(Demon)들을 퇴치하는 비밀 요원 '데몬 헌터'**라는 사실입니다.
   멤버들은 컴백 준비와 월드 투어라는 살인적인 스케줄 속에서도, 틈틈이 출몰하는 악귀들을 처치하며 세상을 구해야 합니다. 화려한 패션과 맛있는 음식, 그리고 멤버들 간의 끈끈한 우정을 바탕으로 악의 세력에 맞서는 이야기를 담고 있습니다.
    ''')

st.divider()

# 2. 등장인물 섹션
st.header('🎭 주요인물')

# 컬럼 레이아웃
char_cols = st.columns(5)

# 캐릭터 정보 리스트
characters = [
    {'name': '루미', 'role': '리더', 'image': 'data/rumi.png'},
    {'name': '미라', 'role': '래퍼', 'image': 'data/mira.png'},
    {'name': '조이', 'role': '래퍼', 'image': 'data/joy.png'}
]

for i, char in enumerate(characters):
    with char_cols[i]:
        # 이미지 출력
        if os.path.exists(char['image']):
            st.image(char['image'], use_container_width=True)
        else:
            st.image(f'https://via.placeholder.com/150x200?text={char["name"]}', 
                    use_container_width=True)
        st.write(f"**{char['name']}**")
        st.caption(char['role'])

st.divider()  # 구분선

# 3. 관련 영상 및 음악
st.header('🎬 관련 미디어')

media_col1, media_col2 = st.columns(2)

with media_col1:
    st.write('#### 📹 관련 영상')
    # 텍스트 입력
    youtube_url = st.text_input('https://www.youtube.com/watch?v=7vCK0VBuQLs&list=RD7vCK0VBuQLs&start_radio=1', 
                                placeholder='https://www.youtube.com/watch?v=...')
    if youtube_url:
        # 동영상 출력
        st.video(youtube_url)

st.divider()  # 구분선

# 데이터 로드
st.header('📊 데이터 분석')

# 데이터 로드 시도
try:
    df = load_data()
    data_loaded = True
    st.success(f'데이터 로드 완료: 총 {len(df)}개의 기사')
except FileNotFoundError:
    st.warning('⚠️ 데이터 파일이 없습니다. data.py를 먼저 실행하세요.')
    data_loaded = False
    
    # 샘플 데이터 생성 (테스트용)
    st.info('테스트용 샘플 데이터를 생성합니다.')
    
    np.random.seed(42)
    dates = pd.date_range(start='2025-06-15', end='2025-09-20', freq='D')
    
    # 샘플 데이터
    sample_data = []
    keywords = ['노래', '케이팝', '한국', '넷플릭스', '인기', '응원', '최고', '문화', '주말', '아이돌', '케데헌', '케데헌 효과']
    
    for date in dates:
        n_articles = np.random.randint(50, 300)
        for _ in range(n_articles // 10):
            title = f"케이팝 데몬 헌터스 {np.random.choice(keywords)} 화제"
            desc = f"{np.random.choice(keywords)} {np.random.choice(keywords)} 케이팝 데몬 헌터스 {np.random.choice(keywords)}"
            sample_data.append({
                'pubDate': date,
                'title': title,
                'description': desc,
                'date': date
            })
    
    df = pd.DataFrame(sample_data)
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['date'] = pd.to_datetime(df['date'])
    data_loaded = True

# 원본 데이터 표시
if data_loaded and show_raw_data:
    st.subheader('📋 원본 데이터')
    st.dataframe(df.head(20))

# 지표 표시
if data_loaded:
    st.subheader('📈 주요 지표')
    
    # 컬럼 레이아웃
    col1, col2, col3 = st.columns(3)
    
    # 지표
    col1.metric("총 기사 수", f"{len(df):,}개")
    col2.metric("분석 기간", f"{(df['date'].max() - df['date'].min()).days}일")
    col3.metric("일평균 기사", f"{len(df) / max((df['date'].max() - df['date'].min()).days, 1):.1f}개")
    
    st.divider()

# AI
# 시계열 분석 (Plotly)
if data_loaded and '시계열 분석' in analysis_options:
    st.header('📈 시계열 분석: 뉴스 기사 수 추이')
    st.write('> 시간에 따른 뉴스 기사 수 변화를 통해 **관심도 추이**와 **주요 이벤트**를 파악')
    
    # 일별 기사 수 집계
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    # 주차 정보 추가
    daily_counts['week'] = daily_counts['date'].dt.isocalendar().week
    daily_counts['month'] = daily_counts['date'].dt.month
    
    # Plotly 그래프
    fig = go.Figure()
    
    # 개봉 후 (6월_영화 개봉달)
    mask1 = daily_counts['month'] == 6
    fig.add_trace(go.Scatter(
        x=daily_counts[mask1]['date'],
        y=daily_counts[mask1]['count'],
        mode='lines+markers',
        name='개봉 후',
        line=dict(color='orange'),
        marker=dict(size=6)
    ))
    
    # 한달 후 (7월)
    mask2 = daily_counts['month'] == 7
    fig.add_trace(go.Scatter(
        x=daily_counts[mask2]['date'],
        y=daily_counts[mask2]['count'],
        mode='lines+markers',
        name='한달 후',
        line=dict(color='green'),
        marker=dict(size=6)
    ))
    
    # 두달 이상 (8~9월)
    mask3 = daily_counts['month'] >= 8
    fig.add_trace(go.Scatter(
        x=daily_counts[mask3]['date'],
        y=daily_counts[mask3]['count'],
        mode='lines+markers',
        name='두달 이상',
        line=dict(color='coral'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='케이팝 데몬 헌터스 뉴스 기사 수 추이',
        xaxis_title='날짜',
        yaxis_title='뉴스 수',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Plotly 차트 출력
    st.plotly_chart(fig, use_container_width=True)
    
    # 해석
    with st.expander('📝 시계열 분석 해석'):
        st.write('''
        **분석 결과:**
        - **개봉달(6월)**: 작품 개봉과 함께 급격한 관심 상승
        - **한달 후(7월)**: OST 빌보드 차트 진입으로 안정적 관심 유지
        - **두달 이상(8~9월)**: 글로벌 시청 기록 달성으로 재확산
        ''')
    
    st.divider()

# 키워드 추이 분석 (Altair)
if data_loaded and '키워드 추이 분석' in analysis_options:
    st.header('📊 주요 키워드 주차별 언급 추이')
    st.write('> 시간에 따른 **주요 키워드의 언급 빈도 변화**를 분석')
    
    # 형태소 분석기
    okt = Okt()
    
    # 주차별 키워드 빈도 계산
    df['week_label'] = df['date'].dt.strftime('%m월 ') + ((df['date'].dt.day - 1) // 7 + 1).astype(str) + '주차'
    
    # 타겟 키워드
    target_keywords = ['노래', '케이팝', '한국', '주말', '넷플릭스', '문화', '인기', '응원', '최고', '케데헌 효과']
    
    # 주차별 키워드 빈도 집계
    keyword_data = []
    
    for week_label in df['week_label'].unique():
        week_df = df[df['week_label'] == week_label]
        all_text = ' '.join(week_df['title'].tolist() + week_df['description'].tolist())
        
        # 텍스트 정제
        cleaned_text = cleanString(all_text)
        
        # 형태소 분리
        words = okt.morphs(cleaned_text)
        
        # 불용어 제거
        words = [word for word in words if word not in stop_words]
        
        # 단어 빈도 계산
        word_counts = Counter(words)
        
        for keyword in target_keywords:
            keyword_data.append({
                'week': week_label,
                '키워드': keyword,
                '빈도': word_counts.get(keyword, 0)
            })
    
    keyword_df = pd.DataFrame(keyword_data)
    
    # Altair 그래프
    chart = alt.Chart(keyword_df).mark_line(point=True).encode(
        x=alt.X('week:N', title='주차', sort=None),
        y=alt.Y('빈도:Q', title='빈도'),
        color=alt.Color('키워드:N', legend=alt.Legend(title='키워드')),
        tooltip=['week', '키워드', '빈도']
    ).properties(
        title='주요 키워드 주차별 언급 추이',
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # 해석
    with st.expander('📝 키워드 추이 분석 해석'):
        st.write('''
        **분석 결과:**
        - **노래, 케이팝**: 작품의 핵심 요소로 지속적으로 높은 언급량
        - **한국, 문화**: K-컬처 관련 담론 형성
        - **인기, 응원**: 팬덤 활동과 관련된 키워드
        ''')
    
    st.divider()

# 키워드 빈도 분석 (Seaborn)
if data_loaded and '키워드 빈도 분석' in analysis_options:
    st.header('🔤 키워드 빈도 분석')
    st.write('> 전체 기간 동안 가장 많이 언급된 **상위 키워드**를 분석')
    
    # 형태소 분석기 (강의록 13.ipynb)
    okt = Okt()
    
    # 전체 텍스트 결합
    all_text = ' '.join(df['title'].tolist() + df['description'].tolist())
    
    # 텍스트 정제 (강의록 13.ipynb)
    cleaned_text = cleanString(all_text)
    
    # 명사 추출 (강의록 13.ipynb)
    nouns = okt.nouns(cleaned_text)
    
    # 불용어 제거 및 한 글자 제거 (강의록 14.ipynb)
    nouns = [word for word in nouns if (len(word) > 1) and (word not in stop_words)]
    
    # 단어 빈도 계산 (강의록 13.ipynb)
    word_counts = Counter(nouns)
    top_words = word_counts.most_common(20)
    
    # 데이터프레임 생성
    word_df = pd.DataFrame(top_words, columns=['단어', '빈도'])
    
    # Seaborn 그래프 (강의록 12.ipynb - st.pyplot)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 색상 설정
    if chart_theme == '다크':
        plt.style.use('dark_background')
        palette = 'rocket'
    elif chart_theme == '컬러풀':
        palette = 'Set2'
    else:
        palette = 'viridis'
    
    sns.barplot(data=word_df, x='빈도', y='단어', palette=palette, ax=ax)
    ax.set_title('상위 20개 키워드 빈도', fontsize=14)
    ax.set_xlabel('빈도')
    ax.set_ylabel('키워드')
    
    # pyplot 출력
    st.pyplot(fig)
    
    # 해석
    with st.expander('📝 키워드 빈도 분석 해석'):
        top3 = [w[0] for w in top_words[:3]]
        st.write(f'''
        **분석 결과:**
        - 가장 많이 언급된 키워드: **{', '.join(top3)}**
        ''')
    
    st.divider()

# 워드클라우드
if data_loaded and '워드클라우드' in analysis_options:
    st.header('☁️ 워드클라우드')
    st.write('> 키워드 빈도를 표현. 글자가 클수록 자주 등장한 키워드.')
    
    # 형태소 분석기
    okt = Okt()
    
    # 전체 텍스트 결합
    all_text = ' '.join(df['title'].tolist() + df['description'].tolist())
    
    # 텍스트 정제
    cleaned_text = cleanString(all_text)
    
    # 명사 추출
    nouns = okt.nouns(cleaned_text)
    
    # 불용어 제거
    nouns = [word for word in nouns if (len(word) > 1) and (word not in stop_words)]
    
    # 워드클라우드용 텍스트
    text_for_wc = ' '.join(nouns)
    
    # 워드클라우드 생성
    # 배경색 설정
    if chart_theme == '다크':
        bg_color = 'black'
    else:
        bg_color = 'white'
    
    # 컬러맵 설정
    if chart_theme == '컬러풀':
        cmap = 'Set3'
    else:
        cmap = 'viridis'
    
    # Pretendard 폰트 경로 사용
    # WordCloud 객체 생성
    wc = WordCloud(
        font_path='font/Pretendard-Regular.ttf',
        max_words=top_n_words,  # 최대 단어 수
        width=800,
        height=400,
        background_color=bg_color,
        colormap=cmap,
        random_state=42
    ).generate(text_for_wc)
    
    # 워드클라우드 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('케이팝 데몬 헌터스 키워드 워드클라우드', fontsize=16, pad=20)
    
    # pyplot 출력
    st.pyplot(fig)
    
    # 해석
    with st.expander('📝 워드클라우드 해석'):
        st.write('''
        **분석 결과:**
        - 중앙에 크게 표시된 단어들이 핵심 키워드
        ''')
    
    st.divider()

# 네트워크 분석
if data_loaded and '네트워크 분석' in analysis_options:
    st.header('🕸️ 키워드 네트워크 분석')
    st.write('> 키워드 간의 **연관성**을 네트워크로 시각화. 함께 자주 등장하는 키워드들이 연결')
    
    # 형태소 분석기
    okt = Okt()
    
    # 각 기사별 명사 추출
    all_nouns = []
    descriptions = df['description'].tolist()
    
    for text in descriptions:
        # 정제
        text_cleaned = re.sub(r'[^가-힣\s]', '', str(text))
        # 명사 추출
        nouns = okt.nouns(text_cleaned)
        # 불용어 제거
        nouns = [word for word in set(nouns) if (len(word) > 1) and (word not in stop_words)]
        all_nouns.append(nouns)
    
    # 엣지 리스트 생성
    edge_list = []
    for nouns in all_nouns:
        if len(nouns) >= 2:
            # 조합 생성
            pairs = list(combinations(nouns, 2))
            edge_list.extend(pairs)
    
    # 엣지 빈도 계산
    edge_counts = Counter(edge_list)
    
    # 최소 연결 강도 이상인 엣지만 선택
    filtered_edges = [(u, v, w) for (u, v), w in edge_counts.items() if w >= network_min_weight]
    
    if len(filtered_edges) > 0:
        # 그래프 객체 생성
        G = nx.Graph()
        
        # 엣지 추가
        for u, v, w in filtered_edges:
            G.add_edge(u, v, weight=w)
        
        # 노드가 너무 많으면 상위 노드만 선택
        if len(G.nodes()) > 50:
            degree_dict = dict(G.degree())
            top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:50]
            G = G.subgraph(top_nodes).copy()
        
        # 네트워크 시각화
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 레이아웃 생성
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 노드 크기 설정
        node_sizes = [G.degree(n) * 50 for n in G.nodes()]
        
        # 엣지 두께 설정
        edge_widths = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]
        
        # 노드 색상
        if chart_theme == '컬러풀':
            node_color = 'lightcoral'
        else:
            node_color = 'skyblue'
        
        # 그래프 그리기
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_size=node_sizes,
            width=edge_widths,
            font_family='Pretendard',
            font_size=12,
            node_color=node_color,
            edge_color='gray',
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title('케이팝 데몬 헌터스 키워드 네트워크', fontsize=20)
        ax.axis('off')
        
        # pyplot 출력
        st.pyplot(fig)
        
        # 중심성 분석
        st.subheader('📊 중심성 분석')
        
        # 컬럼 레이아웃
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('**연결 중심성**')
            st.caption('많은 키워드와 연결된 핵심 키워드')
            
            # 연결 중심성
            degree_centrality = nx.degree_centrality(G)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            degree_df = pd.DataFrame(top_degree, columns=['키워드', '중심성'])
            st.dataframe(degree_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write('**매개 중심성**')
            st.caption('다른 키워드들을 연결')
            
            # 매개 중심성
            betweenness_centrality = nx.betweenness_centrality(G)
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            between_df = pd.DataFrame(top_betweenness, columns=['키워드', '중심성'])
            st.dataframe(between_df, use_container_width=True, hide_index=True)
        
        # 해석
        with st.expander('📝 네트워크 분석 해석'):
            st.write('''
            **분석 결과:**
            - **연결 중심성**이 높은 키워드는 가장 많은 다른 키워드와 함께 언급됨을 의미
            - **매개 중심성**이 높은 키워드는 서로 다른 주제들을 연결함을 의미
            ''')
    else:
        # 에러 메시지
        st.error('⚠️ 연결 강도 조건을 만족하는 엣지가 없습니다. 사이드바에서 최소 연결 강도를 낮춰보세요.')
    
    st.divider()


# 종합 결론
st.header('📋 종합 결론')

# 확장 레이아웃 
with st.expander('🔍 팬덤 형성 핵심 요인 분석 결과', expanded=True):
    st.markdown('''
    ### 케이팝 데몬 헌터스 팬덤 형성 핵심 요인
    
    본 분석을 통해 다음과 같은 **팬덤 형성의 핵심 요인**을 도출하였습니다:
    
    #### 1️⃣ 콘텐츠 요인
    - 독특한 K-POP + 판타지 장르 결합
    - 아이돌 출연진의 연기력과 스타성
    - 몰입감 있는 스토리라인
    
    #### 2️⃣ 미디어 노출 요인
    - 넷플릭스 글로벌 플랫폼을 통한 동시 공개
    - SNS를 통한 바이럴 확산
    - 지속적인 언론 보도
    
    #### 3️⃣ 음악 요인
    - OST 빌보드 차트 진입 (Hot 100)
    - 음원 차트 역주행
    - K-POP 팬덤과의 시너지
    
    #### 4️⃣ 글로벌 요인
    - 다국어 자막 지원
    - 글로벌 시청 기록 달성
    - 해외 팬덤 형성
    ''')

st.divider()  # 구분선
st.caption('🎵 K팝 데몬 헌터스 팬덤 분석 대시보드 | C221088 최유빈 | 2025 데이터시각화')
