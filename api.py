# 네이버 API를 이용한 데이터 수집
import os
import sys
import my_apikeys as mykeys
import urllib.request
import pandas as pd
import json

# 네이버에서 발급받은 클라이언트 ID와 시크릿을 사용
client_id = mykeys.client_id
client_secret = mykeys.client_secret

# 파라미터 설정
display_count = 100     # 한 페이지에 표시할 검색 결과 수
num_data = 1000         # 검색할 데이터 개수
sort = 'date'           # 정렬 기준 (date: 날짜순, sim: 유사도순)

# 검색할 단어와 URL 설정
encText = urllib.parse.quote("서울시 부동산")

# 결과를 저장할 list 생성
results = []

# for문을 사용하여 검색 결과를 페이지별로 요청
for idx in range(1, num_data + 1, display_count):

    # JSON 결과 요청 URL 생성
    url = (
        "https://openapi.naver.com/v1/search/news?query="
        + encText
        + f"&start={idx}&display={display_count}&sort={sort}"
    )

    # 요청 객체 생성
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    # 요청 보내서 응답 받기
    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if rescode == 200:  # 응답 코드가 200이면 성공
        # 응답 본문을 읽음
        response_body = response.read()
        # response_body는 바이트 문자열이므로 decode를 통해 문자열로 변환
        response_dict = json.loads(response_body.decode('utf-8'))
        # dictionary에서 'items' 키를 사용하여 뉴스 기사 목록을 가져옴
        results = results + response_dict['items']
    else:
        print("Error Code:", rescode)

# 데이터 개수 확인
print(f"총 데이터 개수: {len(results)}")

# 일부 데이터 출력
results[:3]
