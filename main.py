"""
Mini NPU Simulator
AI가 계산하는 방식을 흉내 내는 작은 계산기 (MAC 연산)
"""

import json
import os
import re
import time

EPS = 1e-9
REPEAT = 10
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 모드 1 / 성능 분석용 3x3 기본 필터·패턴
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
    total = 0.0
    n = len(pattern)
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


def normalize_label(raw):
    """입력 라벨을 표준 라벨(Cross / X)로 정규화한다."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    return mapping.get(key)


def decide(score_a, score_b, label_a="A", label_b="B", undecided="UNDECIDED"):
    """
    epsilon 기반 점수 비교.
    |score_a - score_b| < EPS 이면 동점(undecided).
    """
    if abs(score_a - score_b) < EPS:
        return undecided
    return label_a if score_a > score_b else label_b


def shapes_ok(matrix, n):
    """n×n 행렬인지 검증한다."""
    if not isinstance(matrix, list) or len(matrix) != n:
        return False
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
        for _ in range(n):
            try:
                line = input()
            except EOFError:
                ok = False
                break
            parts = line.split()
            if len(parts) != n:
                ok = False
                # 남은 줄이 있어도 무시하고 재입력 유도
                break
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                ok = False
                break
        if ok and len(rows) == n:
            return rows
        print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")


def measure_mac(pattern, filt, repeats=REPEAT):
    """MAC 연산만 반복 측정해 평균 시간(ms)을 반환한다 (I/O 제외)."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filt)
    elapsed = time.perf_counter() - t0
    return (elapsed / repeats) * 1000.0


def print_perf_table(rows):
    """크기 / 평균 시간(ms) / 연산 횟수(N²) 표 출력."""
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")
    for size, avg_ms, ops in rows:
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
    avg_ms = measure_mac(pattern, filter_a)
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
    """패턴 키 'size_13_1'에서 N=13을 추출한다."""
    m = re.match(r"size_(\d+)_\d+", key)
    return int(m.group(1)) if m else None


def load_filters(data):
    """
    filters.size_N 의 cross/x 를 표준 라벨로 정규화해 로드한다.
    반환: {5: {"Cross": [...], "X": [...]}, ...}
    """
    loaded = {}
    filters = data.get("filters", {})
    for size_key, pair in filters.items():
        m = re.match(r"size_(\d+)$", size_key)
        if not m:
            print(f"! 알 수 없는 필터 키 무시: {size_key}")
            continue
        n = int(m.group(1))
        normalized = {}
        ok = True
        for raw_label, matrix in pair.items():
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


def mode2_json_analysis():
    """모드 2: data.json 로드 → 케이스 판정 → 성능 분석 → 요약."""
    print()
    print("#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    if not os.path.isfile(DATA_PATH):
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
        total += 1
        case = patterns[case_id]
        print(f"--- {case_id} ---")

        n = parse_size_from_key(case_id)
        if n is None:
            failed += 1
            reason = "키에서 크기를 추출할 수 없음"
            print(f"판정: - | expected: - | FAIL ({reason})")
            fail_list.append(f"{case_id}: {reason}")
            continue

        if n not in filters_by_size:
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
