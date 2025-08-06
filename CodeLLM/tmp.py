from typing import List, Tuple

Point = Tuple[float, float]

def shoelace_area(v: List[Point]) -> float:
    n = len(v)
    a = 0.0
    for i in range(n - 1):
        a += v[i][0] * v[i + 1][1] - v[i + 1][0] * v[i][1]
    a += v[n - 1][0] * v[0][1] - v[0][0] * v[n - 1][1]
    return abs(a) / 2.0

def main():
    v = [(3, 4), (5, 11), (12, 8), (9, 5), (5, 6)]
    area = shoelace_area(v)
    print(f"Given a polygon with vertices {v},")
    print(f"its area is {area}.")

if __name__ == "__main__":
    main()