"""
Mini NPU Simulator
AI가 계산하는 방식을 흉내 내는 작은 계산기 (MAC 연산)
"""

import json
import os
import re      #파이썬에 기본으로 포함된 졍규표현식(문자열 패턴 검색) 모듈임
import time    #time은 경과시간측정, 대기 같은 작업용  datetime은 날짜와 시간 표현 및 계산 작업용

EPS = 1e-9     #1 × 10⁻⁹ = 0.000000001, 현재 코드에서는 아주 작은 오차로 decide 함수는 두 점수의 차이가 0.000000001보다 작으면 부동소수점 오차 등을 고려해 동점으로 판정
REPEAT = 10
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
#__file__은 현재 실행 중인 파이썬 파일의 경로, os.path.abspath()는 절대경로로 변환
#os.path.dirname(...) — 파일 경로에서 폴더 경로만 추출, os.path.join(..., "data.json") — 그 폴더 경로에 data.json을 붙임


# 모드 1 / 성능 분석용 3x3 기본 필터·패턴 / 십자가와 X자 필터
CROSS_3 = [
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
]
X_3 = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
]


def mac(pattern, filt):
    """패턴과 필터를 위치별 곱한 뒤 모두 더한다 (Multiply-Accumulate)."""
    #여기에서 패개변수는 모두 nxn행렬 즉, 2차원 배열의 값이다. 
    total = 0.0
    n = len(pattern)   #len(리스트)는 가장 바깥쪽 리스트에 들어 있는 항목 수를 반환, 행렬에서는 행의 수
    for i in range(n):       #이중for문을 이용한 MAC연산, 예: 구구단
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


def normalize_label(raw):
    """입력 라벨 즉, 다양하게 입력되는 값을 표준 라벨(Cross / X)로 정규화(정해진 값으로 바꿈)한다."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    return mapping.get(key)     #딕셔너리의 .get(key)는 그 키에 저장된 값을 반환합니다. 예: mapping.get("cross") → "Cross", 없는 키(예: apping.get("circle") → None 반환 또는 없는 키일 때 다른 값을 원하면 .get(key, 기본값)처럼 쓸 수도 있어요. 예를 들어 mapping.get("circle", "알 수 없음")은 "알 수 없음"을 반환


def decide(score_a, score_b, label_a="A", label_b="B", undecided="UNDECIDED"):
    """
    epsilon(허용오차) 기반 점수 비교.
    |score_a - score_b| < EPS 이면 동점(undecided).
    """
    if abs(score_a - score_b) < EPS:
        return undecided
    return label_a if score_a > score_b else label_b  #조건표현식 또는 삼항연산자라고 한다. 


def shapes_ok(matrix, n):
    """n×n 행렬인지 검증한다."""
    #early return 방식으로 문제가 있으면 바로 중단하고 마지막 까지 살아남으면 유효하다고 판단. 잘못된 입력을 빠르게 걸러냄.
    if not isinstance(matrix, list) or len(matrix) != n:  #isinstance(matrix, list): matrix가 리스트인지 확인
        return False               #or는 앞 조건이 참이면 뒤 조건을 검사하지 않음
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            return False
    return True


def read_matrix(n, title):
    """콘솔에서 n×n 행렬을 한 줄씩(공백 구분) 입력받는다. 오류 시 재입력."""
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분)")
        rows = []
        ok = True
        for _ in range(n):       #_는 “반복할 때 생기는 값을 사용하지 않는다”는 관례적인 변수 이름
            try:
                line = input()
            except EOFError:   #EOFError는 입력의 끝 종료 표시 = Ctrl+D 즉, 터미널에서 더이상 입력이 없다는 표시. 보통 보통 실행 중인 프로그램을 강제로 중단하는 Ctrl+C와는 다름.
                ok = False
                break          #break: 현재 반복문을 완전히 빠져나옵니다. continue: 현재 반복에서 남은 코드를 건너뛰고 다음 반복으로 넘어갑니다.
            parts = line.split()  #문자열은 split()메서드를 가지고 있으며, line.split()은 그 문자열을 공백 기준으로 나눈 새 리스트를 반환
            if len(parts) != n:
                ok = False
                # 남은 줄이 있어도 무시하고 재입력 유도
                break
            try:
                rows.append([float(x) for x in parts])   #[float(x) for x in parts]는 문자열 리스트의 값을 float값으로 바꾸어 리스트에 재저장, 이후에 rows.append()함.
            except ValueError:
                ok = False
                break
        if ok and len(rows) == n:
            return rows          #return은 함수의 종료임.
        print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")  #for문에서 break될 때 마다 실행 됨.


def measure_mac(pattern, filt, repeats=REPEAT):
    """MAC 연산만 반복 측정해 평균 시간(ms)을 반환한다 (I/O 제외)."""
    t0 = time.perf_counter()  #time모듈의 perf_counter()함수는 경과 시간을 계산하기 위한 정밀한 타이머 값임.
    for _ in range(repeats):
        mac(pattern, filt)
    elapsed = time.perf_counter() - t0 #이런 식으로 경과된 값을 구한다.
    return (elapsed / repeats) * 1000.0  #elapsed = time.perf_counter() - t0는 초 단위로 반환하기에 1000을 곱해서 ms단위로 반환, 즉 1s*1000=1000ms. perf는 performance의 줄임말


def print_perf_table(rows):
    """
    3개의 값을 포함한 튜플이 리스트에 들어있음
    rows = [
        (100, 1.234, 10000),
        (200, 4.567, 40000),
    ]
    """
    for size, avg_ms, ops in rows:
        print(size, avg_ms, ops)
        """크기 / 평균 시간(ms) / 연산 횟수(N²) 표 출력."""
        print("크기       평균 시간(ms)    연산 횟수")
        print("-------------------------------------")
        for size, avg_ms, ops in rows:              #index가 나오는 것은 enumerate(rows)로 해야 함. 여기서는 unpacking(묶음 안의 값을 여러 변수에 각각 나눠 담는 것)을 이용해 바로 size, avg_ms, ops로 받음.
            print(f"{size:<10}{avg_ms:>10.3f}{ops:>12}")


def mode1_user_input():
    """모드 1: 3×3 필터 A/B와 패턴을 콘솔로 입력해 판정한다."""
    n = 3
    print()
    print("#----------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")
    filter_a = read_matrix(n, "필터 A")
    print()
    filter_b = read_matrix(n, "필터 B")
    print()
    print("저장 확인")
    print("필터 A:", filter_a)
    print("필터 B:", filter_b)
    print()

    print("#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")
    pattern = read_matrix(n, "패턴")
    print()

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_mac(pattern, filter_a)   #filter_b를 측정하지 않고 filter_a만 측정하는 이유는 두 필터의 크기가 같고 MAC연산이 동일한 연산이므로, 한 번만 측정해도 충분하다고 판단했기 때문임.
    verdict = decide(score_a, score_b, label_a="A", label_b="B", undecided="판정 불가")

    print("#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{REPEAT}회): {avg_ms:.3f} ms")
    if verdict == "판정 불가":
        print(f"판정: 판정 불가 (|A-B| < {EPS})")
    else:
        print(f"판정: {verdict}")
    print()

    print("#---------------------------------------")
    print("# [4] 성능 분석 (3×3, 평균/{REPEAT}회)")
    print("#---------------------------------------")
    print_perf_table([("3×3", avg_ms, n * n)])


def parse_size_from_key(key):
    """패턴 키 'size_13_1'에서 N=13을 추출한다. 즉, data.json에서 정규표현식을 이용해서 patterns 키에서 size_N_xxx에서 N을 추출하는 용도임."""
    m = re.match(r"size_(\d+)_\d+", key)
    return int(m.group(1)) if m else None
    """
    m.group(0)  # "size_13_1" 전체 매치
    m.group(1)  # "13" 즉, 괄호로 캡처한 첫 번째 그룹

    """


def load_filters(data):
    """
    data.json의 filters.size_N 의 cross/x 를 표준 라벨로 정규화해 로드한다.
    그리고 마지막에는 다음과 같이 반환: {5: {"Cross": [...], "X": [...]}, ...}
    """
    loaded = {}
    filters = data.get("filters", {})
    for size_key, pair in filters.items():   # for in 문은 for가 in 문은 in 뒤에 놓이는 시퀀스(리스트, 튜플, 문자열 등)의 각 항목을 순회하며 반복하는 구문입니다. 여기서는 filters.items()는 dict_items라는 객체로 딕셔너리의 키와 값 쌍이 튜플로 나열 되어 있음.
        """
        filters.items()는 즉, 딕셔너리값인 filters에서 키와 값 쌍을 하나씩 꺼내는 기능입니다.
        
        filters = {
            "size_3": {"cross": [...], "x": [...]},
            "size_5": {"cross": [...], "x": [...]},
        }에서

        ("size_3", {"cross": [...], "x": [...]})
        ("size_5", {"cross": [...], "x": [...]})
        값을 튜플로 꺼낸다.
        """
        m = re.match(r"size_(\d+)$", size_key)  #정규표현식에서 $은 문자열의 끝을 의미함. 즉, size_숫자 형태만 허용하고 size_숫자_숫자 형태는 허용하지 않음.
        if not m:
            print(f"! 알 수 없는 필터 키 무시: {size_key}")
            continue
        n = int(m.group(1))
        normalized = {}
        ok = True
        for raw_label, matrix in pair.items():  
            """
            pair = {"cross": [...], "x": [...]}에서
            ("cross", [...])
            ("x", [...])
            값을 튜플로 꺼낸다.
            """
            label = normalize_label(raw_label)
            if label is None:
                print(f"! {size_key}: 알 수 없는 필터 라벨 '{raw_label}'")
                ok = False
                continue
            if not shapes_ok(matrix, n):
                print(f"! {size_key}/{raw_label}: 크기 불일치 (기대 {n}×{n})")
                ok = False
                continue
            normalized[label] = matrix
        if "Cross" in normalized and "X" in normalized:
            loaded[n] = normalized
            print(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
        elif ok:
            print(f"! {size_key}: Cross/X 필터가 모두 없어 건너뜀")
    return loaded
    """
    loaded가 딕셔너리 형태로 아래와 같음
        {
        3: {
            "Cross": [[...], [...], [...]],
            "X":     [[...], [...], [...]]
        },
        5: {
            "Cross": [[...], ...],
            "X":     [[...], ...]
        }
    }
    """


def mode2_json_analysis():
    """모드 2: data.json 로드 → 케이스 판정 → 성능 분석 → 요약."""
    print()
    print("#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    if not os.path.isfile(DATA_PATH):     #지정된 경로에 존재하는 것이 일반 파일인지 확인, 일반 파일이면 True 나머지 경우는 False
        print(f"오류: data.json을 찾을 수 없습니다 ({DATA_PATH})")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    filters_by_size = load_filters(data)
    patterns = data.get("patterns", {})

    print()
    print("#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    total = 0
    passed = 0
    failed = 0
    fail_list = []
    # 성능 측정용: 크기별 (pattern, cross_filter) 하나 보관
    samples = {3: (CROSS_3, CROSS_3)}

    for case_id in sorted(patterns.keys(), key=lambda k: (parse_size_from_key(k) or 0, k)):
        """
        sorted()는 반복 가능한 자료형의 값들을 정렬해서 새 리스트로 반환하는 파이썬 내장 함수로 기본적으로 숫자는 작은 순서대로, 문자열은 사전식 순서대로 정렬
        numbers = [3, 1, 2]
        result = sorted(numbers)
        print(result)   # [1, 2, 3]
   
        sorted(정렬할_대상, key=정렬_기준)으로, 
        첫 번째 인자: 정렬할 대상, key에 전달하는 값: 각 항목의 정렬 기준을 계산하는 함수로 정렬할 대상들을 key 함수에 전달하여 반환값을 기준(숫자면 작은 순, 문자면 사전식 순)으로 정렬할 대상을 정렬하는 방식
        그런데 만약 key=정렬 기준이 두 개 앞은 첫번째 기준이 되고 두 번째는 첫 번째 기준이 같을 때 두 번째 기준으로 정렬하는 방식으로 사용 가능 

        patterns.keys()는 data.json의 patterns 키에 있는 모든 케이스 키를 가져옴
        key=lambda k: (parse_size_from_key(k) or 0, k)에서 lambda는 익명 함수(이름 없는 함수)로, 매개변수(k를) 입력받아 : 계산식(parse_size_from_key(k) or 0, k) 계산하여 반환하는 함수임. 
         ( a or 0, "size_13_1")에서 or는 왼쪽 값이 None 같은 “거짓으로 취급되는 값”이면 오른쪽 값을 사용하기에 a에 거짓으로 취급되는 값이 오면 or 때문에만약 추출 실패 시 0으로 처리하여 정렬 기준으로 삼음. 그리고 처음 기준이 같으면 두 번째 기준으로는 케이스 키 자체를 사용하여 사전식 순서로 정렬함.
        """
        total += 1
        case = patterns[case_id]
        print(f"--- {case_id} ---")
        """
                case = {
            "input": [
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0]
            ],
            "expected": "x"
        }
        """

        n = parse_size_from_key(case_id)
        if n is None:
            failed += 1
            reason = "키에서 크기를 추출할 수 없음"
            print(f"판정: - | expected: - | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")
            continue

        if n not in filters_by_size:   # in 또는 not in 뒤에 딕셔너리가 오면 기본적으로 딕셔너리의 키를 확인
            failed += 1
            reason = f"size_{n} 필터 없음"
            print(f"판정: - | expected: - | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")
            continue

        pattern = case.get("input")
        expected_raw = case.get("expected")
        expected = normalize_label(expected_raw)

        if not shapes_ok(pattern, n):
            failed += 1
            reason = f"패턴 크기 불일치 (기대 {n}×{n})"
            print(f"판정: - | expected: {expected or expected_raw} | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")
            continue

        if expected is None:
            failed += 1
            reason = f"expected 라벨 정규화 실패 ('{expected_raw}')"
            print(f"판정: - | expected: {expected_raw} | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")
            continue

        cross_f = filters_by_size[n]["Cross"]
        x_f = filters_by_size[n]["X"]
        score_cross = mac(pattern, cross_f)
        score_x = mac(pattern, x_f)
        verdict = decide(
            score_cross, score_x, label_a="Cross", label_b="X", undecided="UNDECIDED"
        )

        print(f"Cross 점수: {score_cross}")
        print(f"X 점수: {score_x}")

        if verdict == expected:
            passed += 1
            print(f"판정: {verdict} | expected: {expected} | PASS")
        else:
            failed += 1
            if verdict == "UNDECIDED":
                reason = "동점 규칙"
            else:
                reason = f"판정 불일치 ({verdict} != {expected})"
            print(f"판정: {verdict} | expected: {expected} | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")

        if n not in samples:
            samples[n] = (pattern, cross_f)

    print()
    print("#---------------------------------------")
    print(f"# [3] 성능 분석 (평균/{REPEAT}회)")
    print("#---------------------------------------")
    perf_rows = []
    for n in sorted(samples.keys()):
        pattern, filt = samples[n]
        avg_ms = measure_mac(pattern, filt)
        perf_rows.append((f"{n}×{n}", avg_ms, n * n))
    print_perf_table(perf_rows)

    print()
    print("#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    if fail_list:
        print()
        print("실패 케이스:")
        for item in fail_list:
            print(f"- {item}")
    print()
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')


def main():
    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print()
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    choice = input("선택: ").strip()

    if choice == "1":
        mode1_user_input()
    elif choice == "2":
        mode2_json_analysis()
    else:
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()
