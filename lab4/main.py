"""
Лабораторная работа: Матрицы выборок
- С возвращением: m раз выбираем один из n элементов (повторы возможны)
- Без возвращения: выбираем m различных элементов из n (m ≤ n)
- Размещение m = n: перестановка всех n элементов
- Матрица накопления: результаты нескольких проходов (абсолютные частоты)
Матрица m×n: строка i — i-й выбор, столбец j — элемент j; 1 если выбран
"""
import numpy as np
import random

SEED = 42


def generate_params():
    """Генерирует случайные параметры n и m (3 ≤ m ≤ n ≤ 10)"""
    n = random.randint(3, 10)
    m = random.randint(3, n)
    return n, m


def with_replacement(n, m, rng):
    """
    Выборка с возвращением.
    Возвращает матрицу m×n и список выбранных элементов.
    """
    mat = np.zeros((m, n), dtype=int)
    sample = []
    for i in range(m):
        j = rng.integers(0, n)  # выбираем индекс элемента
        mat[i, j] = 1
        sample.append(j + 1)  # элемент (1..n)
    return mat, sample


def without_replacement(n, m, rng):
    """
    Выборка без возвращения.
    Возвращает матрицу m×n и список выбранных элементов.
    """
    perm = rng.permutation(n)[:m]  # случайная перестановка, берем первые m
    mat = np.zeros((m, n), dtype=int)
    sample = []
    for i, j in enumerate(perm):
        mat[i, j] = 1
        sample.append(j + 1)  # элемент (1..n)
    return mat, sample


def placement_m_n(n, rng):
    """
    Размещение m = n (перестановка всех элементов).
    Возвращает матрицу n×n и список выбранных элементов.
    """
    return without_replacement(n, n, rng)


def accumulation_matrix(n, m, num_passes, rng, with_replacement_flag=True):
    """
    Создает матрицу накопления для нескольких проходов.
    Возвращает матрицу накопления (абсолютные частоты).
    """
    accumulation_mat = np.zeros((m, n), dtype=int)
    
    for _ in range(num_passes):
        if with_replacement_flag:
            for i in range(m):
                j = rng.integers(0, n)
                accumulation_mat[i, j] += 1
        else:
            perm = rng.permutation(n)[:m]
            for i, j in enumerate(perm):
                accumulation_mat[i, j] += 1
    
    return accumulation_mat


def print_table(title, n, m, mat, sample):
    """
    Красиво печатает матрицу с подписями строк и столбцов,
    а также массив 0-1 и саму выборку.
    """
    print(title)
    print(f"n = {n}, m = {m}")
    rows, cols = mat.shape
    
    # Заголовок с номерами столбцов (элементы 1..n)
    header = "    " + " ".join(f"{j+1:>3}" for j in range(cols))
    print(header)
    print("    " + "-" * (4 * cols))
    
    # Строки с номерами шагов выбора
    for i in range(rows):
        print(f"{i+1:>3}|" + " ".join(f"{mat[i, j]:>3}" for j in range(cols)))
    
    # Вывод выборки (последовательность выбранных элементов)
    print(f"\nВыборка (последовательность элементов): {sample}")
    
    # Вывод матрицы в формате 0-1
    print(f"\nМатрица в формате 0-1 (m={m}, n={n}):")
    print(mat.tolist())
    print()


def print_accumulation_table(title, n, m, acc_mat, num_passes):
    """
    Печатает матрицу накопления.
    """
    print(title)
    print(f"n = {n}, m = {m}, количество проходов = {num_passes}")
    
    print("\nМАТРИЦА НАКОПЛЕНИЯ (абсолютные частоты):")
    rows, cols = acc_mat.shape
    header = "    " + " ".join(f"{j+1:>4}" for j in range(cols))
    print(header)
    print("    " + "-" * (5 * cols))
    for i in range(rows):
        print(f"{i+1:>3}|" + " ".join(f"{acc_mat[i, j]:>4}" for j in range(cols)))
    
    # Вывод матрицы накопления в формате списка
    print(f"\nМатрица абсолютных частот (m={m}, n={n}):")
    print(acc_mat.tolist())
    print()

def demonstrate_accumulation(rng):
    """
    Демонстрирует работу с матрицей накопления
    """
    print("\n" + "="*70)
    print("МАТРИЦЫ НАКОПЛЕНИЯ (МНОГОКРАТНЫЕ ПРОХОДЫ)")
    print("="*70)
    
    # Параметры для демонстрации
    n, m = 5, 3
    num_passes_list = [10, 100, 1000]
    
    print(f"\nДемонстрация для n={n}, m={m}")
    print("Сравнение для разного количества проходов:\n")
    
    for with_replacement_flag in [True, False]:
        choice_type = "С ВОЗВРАЩЕНИЕМ" if with_replacement_flag else "БЕЗ ВОЗВРАЩЕНИЯ"
        print(f"\n--- ВЫБОРКА {choice_type} ---")
        
        for num_passes in num_passes_list:
            accumulation_mat = accumulation_matrix(
                n, m, num_passes, rng, with_replacement_flag
            )
            
            print_accumulation_table(
                f"\nРезультаты для {num_passes} проходов:",
                n, m, accumulation_mat, num_passes
            )


def main():
    # Используем современный генератор случайных чисел numpy
    rng = np.random.default_rng(SEED)
    
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА: МАТРИЦЫ ВЫБОРОК")
    print("="*70)
    
    # 1) Демонстрация с фиксированными параметрами
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ С ФИКСИРОВАННЫМИ ПАРАМЕТРАМИ")
    print("="*70)
    
    # Выборки с возвращением
    with_replacement_params = [(5, 4), (7, 5), (8, 6), (6, 4)]
    print("\n--- ВЫБОРКИ С ВОЗВРАЩЕНИЕМ ---")
    for n, m in with_replacement_params:
        mat, sample = with_replacement(n, m, rng)
        print_table(f"С возвращением: n={n}, m={m}", n, m, mat, sample)
    
    # Выборки без возвращения
    without_replacement_params = [(6, 4), (8, 5), (7, 4), (5, 3)]
    print("\n--- ВЫБОРКИ БЕЗ ВОЗВРАЩЕНИЯ ---")
    for n, m in without_replacement_params:
        mat, sample = without_replacement(n, m, rng)
        print_table(f"Без возвращения: n={n}, m={m}", n, m, mat, sample)
    
    # Размещения m = n
    placement_params = [5, 7, 8, 4]
    print("\n--- РАЗМЕЩЕНИЯ m = n (ПЕРЕСТАНОВКИ) ---")
    for n in placement_params:
        mat, sample = placement_m_n(n, rng)
        print_table(f"Размещение m=n: n={n}", n, n, mat, sample)
    
    # 2) Демонстрация со случайными параметрами
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ СО СЛУЧАЙНЫМИ ПАРАМЕТРАМИ")
    print("="*70)
    
    # Генерируем случайные параметры 3 раза
    for trial in range(1, 4):
        print(f"\n--- СЛУЧАЙНЫЙ ТЕСТ #{trial} ---")
        n, m = generate_params()
        print(f"Сгенерированы параметры: n={n}, m={m}")
        
        # С возвращением
        mat_with, sample_with = with_replacement(n, m, rng)
        print_table("С возвращением:", n, m, mat_with, sample_with)
        
        # Без возвращения
        mat_without, sample_without = without_replacement(n, m, rng)
        print_table("Без возвращения:", n, m, mat_without, sample_without)
        
        # Если m == n, показываем как размещение
        if m == n:
            print("⚠️ В этом случае m = n, выборка без возвращения является перестановкой!")
    
    # 3) Демонстрация матриц накопления
    demonstrate_accumulation(rng)
    


if __name__ == "__main__":
    main()



# def print_explanation():
#     """Выводит объяснение различий между типами выборок"""
#     print("\n" + "="*70)
#     print("ОБЪЯСНЕНИЕ РАЗЛИЧИЙ МЕЖДУ ТИПАМИ ВЫБОРОК")
#     print("="*70)
#     
#     print("\n1. ВЫБОРКА С ВОЗВРАЩЕНИЕМ:")
#     print("   • Элемент после выбора возвращается обратно в множество")
#     print("   • На каждом шаге доступны все n элементов")
#     print("   • Элементы могут повторяться в выборке")
#     print("   • Количество возможных выборок: n^m")
#     print("   • В матрице: в каждом столбце может быть несколько единиц")
#     print("   • Математическое ожидание частоты для каждой ячейки: 1/n")
#     
#     print("\n2. ВЫБОРКА БЕЗ ВОЗВРАЩЕНИЯ:")
#     print("   • Выбранный элемент удаляется из множества")
#     print("   • Каждый элемент может быть выбран не более одного раза")
#     print("   • Все элементы в выборке различны")
#     print("   • Количество возможных выборок: n!/(n-m)!")
#     print("   • В матрице: в каждом столбце не более одной единицы")
#     print("   • Математическое ожидание: для i-го шага вероятность = 1/(n-i+1)")
#     
#     print("\n3. РАЗМЕЩЕНИЕ (m = n):")
#     print("   • Частный случай выборки без возвращения")
#     print("   • Выбираются все n элементов в некотором порядке")
#     print("   • Получается перестановка из n элементов")
#     print("   • Количество возможных перестановок: n!")
#     print("   • В матрице: ровно одна единица в каждой строке и каждом столбце")
#     
#     print("\n4. МАТРИЦА НАКОПЛЕНИЯ:")
#     print("   • Результат нескольких проходов (экспериментов)")
#     print("   • Абсолютные частоты показывают, сколько раз элемент был выбран")
#     print("   • Относительные частоты приближаются к теоретическим вероятностям")
#     print("   • При увеличении числа проходов частоты стремятся к вероятностям")
