import numpy as np
import random

def generate_matrices():
    """
    Генерирует две матрицы для выборок с возвращением и без возвращения
    """
    # 1) Генерируем случайные числа m и n в диапазоне от 3 до 10
    n = random.randint(3, 10)  # размер исходного множества
    m = random.randint(3, n)   # количество выбираемых элементов (m ≤ n)
    
    print(f"Исходные параметры:")
    print(f"n = {n} (размер исходного множества, элементы 1...{n})")
    print(f"m = {m} (количество выбираемых элементов)")
    print()
    
    # Создаем исходное множество элементов
    elements = list(range(1, n + 1))
    print(f"Исходное множество: {elements}")
    print()
    
    # 2) Создаем матрицы m×n, заполненные нулями
    matrix_with_return = np.zeros((m, n), dtype=int)
    matrix_without_return = np.zeros((m, n), dtype=int)
    
    # 3) Генерируем выборки и заполняем матрицы
    
    # Выборка с возвращением
    print("ВЫБОРКА С ВОЗВРАЩЕНИЕМ:")
    print("На каждом шаге выбираем элемент, после чего возвращаем его обратно")
    print("(элементы могут повторяться в разных шагах)")
    
    sample_with_return = []
    for i in range(m):
        # Выбираем случайный элемент из всего множества
        chosen_element = random.choice(elements)
        sample_with_return.append(chosen_element)
        
        # В i-й строке ставим 1 в столбце выбранного элемента
        # Индексы в Python начинаются с 0, поэтому вычитаем 1
        matrix_with_return[i, chosen_element - 1] = 1
        
        print(f"Шаг {i+1}: выбран элемент {chosen_element}")
    
    print(f"\nПолученная выборка: {sample_with_return}")
    print("\nМатрица для выборки с возвращением (m×n):")
    print("Строка i - i-й выбор, столбец j - элемент j")
    print("1 означает, что на этом шаге выбран данный элемент")
    print(matrix_with_return)
    print()
    
    # Выборка без возвращения
    print("ВЫБОРКА БЕЗ ВОЗВРАЩЕНИЯ:")
    print("На каждом шаге выбираем элемент и удаляем его из множества")
    print("(все выбранные элементы различны)")
    
    # Создаем копию множества для выборки без возвращения
    available_elements = elements.copy()
    sample_without_return = []
    
    for i in range(m):
        # Выбираем случайный элемент из доступных
        chosen_element = random.choice(available_elements)
        sample_without_return.append(chosen_element)
        available_elements.remove(chosen_element)
        
        # В i-й строке ставим 1 в столбце выбранного элемента
        matrix_without_return[i, chosen_element - 1] = 1
        
        print(f"Шаг {i+1}: выбран элемент {chosen_element}")
        print(f"  Остались доступными: {available_elements}")
    
    print(f"\nПолученная выборка: {sample_without_return}")
    print("\nМатрица для выборки без возвращения (m×n):")
    print("Строка i - i-й выбор, столбец j - элемент j")
    print("1 означает, что на этом шаге выбран данный элемент")
    print(matrix_without_return)
    print()
    
    # Особый случай: если m = n, это перестановка
    if m == n:
        print("⚠️ При m = n получается перестановка всех элементов!")
        print(f"Перестановка: {sample_without_return}")
    
    return matrix_with_return, matrix_without_return, n, m

def explain_difference():
    """
    Объясняет разницу между выборками с возвращением и без
    """
    print("\n" + "="*60)
    print("ОБЪЯСНЕНИЕ РАЗЛИЧИЙ МЕЖДУ ВЫБОРКАМИ")
    print("="*60)
    
    print("\n1. Выборка С ВОЗВРАЩЕНИЕМ:")
    print("   • После каждого выбора элемент возвращается обратно в множество")
    print("   • На каждом шаге все элементы доступны для выбора")
    print("   • Элементы могут повторяться в выборке")
    print("   • Количество возможных выборок: n^m")
    print("   • В матрице: в каждом столбце может быть несколько единиц")
    
    print("\n2. Выборка БЕЗ ВОЗВРАЩЕНИЯ:")
    print("   • Выбранный элемент удаляется из множества")
    print("   • Каждый элемент может быть выбран не более одного раза")
    print("   • Все элементы в выборке различны")
    print("   • Количество возможных выборок: n!/(n-m)!")
    print("   • В матрице: в каждом столбце не более одной единицы")
    print("   • Если m = n, получаем перестановку из n элементов")

def visualize_matrices(matrix_with_return, matrix_without_return, n, m):
    """
    Визуализирует матрицы для наглядности
    """
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ МАТРИЦ")
    print("="*60)
    
    print(f"\nМатрица {m}×{n} для выборки С ВОЗВРАЩЕНИЕМ:")
    print("    " + "  ".join([f"j={j+1}" for j in range(n)]))
    for i in range(m):
        row_str = f"i={i+1}: "
        for j in range(n):
            if matrix_with_return[i, j] == 1:
                row_str += " [1]"
            else:
                row_str += "  0"
        print(row_str)
    
    print(f"\nМатрица {m}×{n} для выборки БЕЗ ВОЗВРАЩЕНИЯ:")
    print("    " + "  ".join([f"j={j+1}" for j in range(n)]))
    for i in range(m):
        row_str = f"i={i+1}: "
        for j in range(n):
            if matrix_without_return[i, j] == 1:
                row_str += " [1]"
            else:
                row_str += "  0"
        print(row_str)

# Запуск программы
if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости (можно убрать для случайности)
    random.seed(42)
    
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА: МАТРИЦЫ ВЫБОРОК")
    print("="*60)
    
    # Генерируем матрицы
    matrix_with_return, matrix_without_return, n, m = generate_matrices()
    
    # Визуализируем матрицы
    visualize_matrices(matrix_with_return, matrix_without_return, n, m)
    
    # Объясняем разницу
    #explain_difference()
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print(f"• Для заданных параметров n={n}, m={m} мы получили две матрицы размером {m}×{n}")
    print("• Матрица с возвращением может иметь несколько единиц в одном столбце")
    print("• Матрица без возвращения имеет не более одной единицы в каждом столбце")
    print("• Это отражает основное различие: возможность повторения элементов")