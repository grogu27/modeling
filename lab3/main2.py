import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Создаем директорию для сохранения рисунков, если её нет
if not os.path.exists('figures'):
    os.makedirs('figures')

# Устанавливаем seed для воспроизводимости
np.random.seed(42)


def generate_doubly_stochastic_matrix(n, max_iter=1000, tol=1e-10):
    """
    Генерация случайной двустохастической матрицы методом Синкхорна
    """
    # Генерируем случайную матрицу с положительными элементами
    # Используем равномерное распределение для большей случайности
    A = np.random.uniform(0, 1, size=(n, n))
    
    # Итеративная нормализация строк и столбцов
    for _ in range(max_iter):
        # Нормализация строк
        row_sums = A.sum(axis=1, keepdims=True)
        A = A / row_sums
        
        # Нормализация столбцов
        col_sums = A.sum(axis=0, keepdims=True)
        A = A / col_sums
        
        # Проверка сходимости
        if np.max(np.abs(row_sums - 1)) < tol and np.max(np.abs(col_sums - 1)) < tol:
            break
    
    return A


def check_doubly_stochastic(P, tol=1e-10):
    """
    Проверка, что матрица является двустохастической
    """
    n = P.shape[0]
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    
    rows_ok = np.all(np.abs(row_sums - 1) < tol)
    cols_ok = np.all(np.abs(col_sums - 1) < tol)
    nonneg_ok = np.all(P >= 0)
    
    return rows_ok and cols_ok and nonneg_ok


def simulate_markov_chain(P, initial_state, n_steps):
    """
    Моделирование цепи Маркова
    """
    n = P.shape[0]
    trajectory = np.zeros(n_steps, dtype=int)
    trajectory[0] = initial_state
    
    # Матрица накопленных вероятностей для быстрого выбора
    cumsum_P = np.cumsum(P, axis=1)
    
    for t in range(1, n_steps):
        current_state = trajectory[t-1]
        r = np.random.random()
        
        # Выбор следующего состояния методом накопленных вероятностей
        trajectory[t] = np.searchsorted(cumsum_P[current_state], r)
    
    return trajectory


def compute_transition_frequencies(trajectory, n_states):
    """
    Вычисление частот переходов
    """
    transition_counts = np.zeros((n_states, n_states))
    
    for t in range(len(trajectory) - 1):
        i = trajectory[t]
        j = trajectory[t+1]
        transition_counts[i, j] += 1
    
    # Нормализация по строкам
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Избегаем деления на ноль
    row_sums[row_sums == 0] = 1
    transition_frequencies = transition_counts / row_sums
    
    return transition_frequencies


def compute_autocorrelation(trajectory, max_lag=50):
    """
    Вычисление автокорреляции для траектории
    """
    n = len(trajectory)
    # Нормализуем траекторию
    trajectory_mean = np.mean(trajectory)
    trajectory_std = np.std(trajectory)
    
    if trajectory_std == 0:
        return np.zeros(max_lag + 1)
    
    normalized_traj = (trajectory - trajectory_mean) / trajectory_std
    
    autocorr = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            # Вычисляем автокорреляцию для заданного лага
            cov = np.sum(normalized_traj[:-lag] * normalized_traj[lag:])
            autocorr[lag] = cov / (n - lag)
    
    return autocorr


def plot_heatmap(matrix, title, filename, cmap='viridis'):
    """
    Построение тепловой карты матрицы
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)
    
    # Добавляем значения в ячейки для небольших матриц
    if matrix.shape[0] <= 8:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")
    
    ax.set_xlabel('Состояние (столбец)')
    ax.set_ylabel('Состояние (строка)')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_trajectory(trajectory, title, filename):
    """
    Построение графика траектории
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time = np.arange(len(trajectory))
    ax.plot(time, trajectory, 'b-', linewidth=0.5, alpha=0.7)
    ax.scatter(time[::50], trajectory[::50], c='red', s=10, alpha=0.5, label='Каждое 50-е состояние')
    
    ax.set_xlabel('Время (шаг)')
    ax.set_ylabel('Состояние')
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_transition_frequencies(freq_matrix, theoretical_matrix, title, filename):
    """
    Построение графика сравнения частот переходов
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Эмпирические частоты
    im1 = axes[0].imshow(freq_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Эмпирические частоты')
    axes[0].set_xlabel('Состояние (в которое)')
    axes[0].set_ylabel('Состояние (из которого)')
    plt.colorbar(im1, ax=axes[0])
    
    # Теоретические вероятности
    im2 = axes[1].imshow(theoretical_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Теоретические вероятности')
    axes[1].set_xlabel('Состояние (в которое)')
    axes[1].set_ylabel('Состояние (из которого)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_autocorrelation_comparison(autocorr1, autocorr2, title, filename):
    """
    Построение графика сравнения автокорреляций
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = np.arange(len(autocorr1))
    
    ax.plot(lags, autocorr1, 'b-', linewidth=2, label='Матрица 1', alpha=0.7)
    ax.plot(lags, autocorr2, 'r-', linewidth=2, label='Матрица 2', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=1.96/np.sqrt(len(autocorr1)), color='gray', linestyle='--', linewidth=1, alpha=0.7, label='95% дов. интервал')
    ax.axhline(y=-1.96/np.sqrt(len(autocorr1)), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Лаг')
    ax.set_ylabel('Автокорреляция')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_state_distribution(trajectory, title, filename):
    """
    Построение гистограммы распределения состояний
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n_states = len(np.unique(trajectory))
    counts = np.bincount(trajectory, minlength=n_states)
    frequencies = counts / len(trajectory)
    
    states = np.arange(n_states)
    ax.bar(states, frequencies, alpha=0.7, color='blue', edgecolor='black', label='Эмпирическое')
    ax.axhline(y=1/n_states, color='red', linestyle='--', linewidth=2, label=f'Теоретическое (1/{n_states})')
    
    ax.set_xlabel('Состояние')
    ax.set_ylabel('Частота')
    ax.set_title(title)
    ax.set_xticks(states)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_behavior_comparison(trajectory1, trajectory2, title, filename):
    """
    Пункт 6: Построение графика сравнения поведения двух цепей
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Первые 500 шагов для обеих цепей
    time = np.arange(min(500, len(trajectory1)))
    
    axes[0].plot(time, trajectory1[:500], 'b-', linewidth=0.7, label='Цепь 1', alpha=0.7)
    axes[0].plot(time, trajectory2[:500], 'r-', linewidth=0.7, label='Цепь 2', alpha=0.7)
    axes[0].set_ylabel('Состояние')
    axes[0].set_title('Сравнение траекторий (первые 500 шагов)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Скользящее среднее (окно 50)
    window = 50
    moving_avg1 = np.convolve(trajectory1, np.ones(window)/window, mode='valid')
    moving_avg2 = np.convolve(trajectory2, np.ones(window)/window, mode='valid')
    time_ma = np.arange(len(moving_avg1))
    
    axes[1].plot(time_ma, moving_avg1, 'b-', linewidth=1.5, label='Цепь 1', alpha=0.8)
    axes[1].plot(time_ma, moving_avg2, 'r-', linewidth=1.5, label='Цепь 2', alpha=0.8)
    axes[1].axhline(y=(trajectory1.shape[0]-1)/2, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Теоретическое среднее')
    axes[1].set_ylabel('Среднее состояние (окно 50)')
    axes[1].set_title('Сравнение скользящих средних')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Гистограммы распределений
    n_states = len(np.unique(np.concatenate([trajectory1, trajectory2])))
    counts1 = np.bincount(trajectory1, minlength=n_states)
    counts2 = np.bincount(trajectory2, minlength=n_states)
    freq1 = counts1 / len(trajectory1)
    freq2 = counts2 / len(trajectory2)
    states = np.arange(n_states)
    
    width = 0.35
    axes[2].bar(states - width/2, freq1, width, alpha=0.7, color='blue', edgecolor='black', label='Цепь 1')
    axes[2].bar(states + width/2, freq2, width, alpha=0.7, color='red', edgecolor='black', label='Цепь 2')
    axes[2].axhline(y=1/n_states, color='green', linestyle='--', linewidth=2, label=f'Теоретическое (1/{n_states})')
    axes[2].set_xlabel('Состояние')
    axes[2].set_ylabel('Частота')
    axes[2].set_title('Сравнение распределений состояний')
    axes[2].set_xticks(states)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


def plot_autocorrelation_detailed(autocorr1, autocorr2, n_steps, title, filename):
    """
    Пункт 7: Построение детального графика автокорреляции для обеих матриц
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    lags = np.arange(len(autocorr1))
    conf_level = 1.96 / np.sqrt(n_steps)
    
    # Индивидуальные графики с доверительными интервалами
    axes[0, 0].bar(lags, autocorr1, width=0.8, color='blue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    axes[0, 0].fill_between(lags, conf_level, -conf_level, color='gray', alpha=0.2, label='95% дов. интервал')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].axhline(y=conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(y=-conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Лаг')
    axes[0, 0].set_ylabel('Автокорреляция')
    axes[0, 0].set_title('Автокорреляционная функция - Цепь 1')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(lags, autocorr2, width=0.8, color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    axes[0, 1].fill_between(lags, conf_level, -conf_level, color='gray', alpha=0.2, label='95% дов. интервал')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].axhline(y=conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 1].axhline(y=-conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 1].set_xlabel('Лаг')
    axes[0, 1].set_ylabel('Автокорреляция')
    axes[0, 1].set_title('Автокорреляционная функция - Цепь 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Сравнение автокорреляций
    axes[1, 0].plot(lags, autocorr1, 'b-o', linewidth=2, markersize=4, label='Цепь 1', alpha=0.7)
    axes[1, 0].plot(lags, autocorr2, 'r-s', linewidth=2, markersize=4, label='Цепь 2', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].axhline(y=conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 0].axhline(y=-conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Лаг')
    axes[1, 0].set_ylabel('Автокорреляция')
    axes[1, 0].set_title('Сравнение автокорреляционных функций')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Разность автокорреляций
    autocorr_diff = autocorr1 - autocorr2
    axes[1, 1].bar(lags, autocorr_diff, width=0.8, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].axhline(y=2*conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='2σ граница')
    axes[1, 1].axhline(y=-2*conf_level, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Лаг')
    axes[1, 1].set_ylabel('Разность автокорреляций (Цепь 1 - Цепь 2)')
    axes[1, 1].set_title('Разность автокорреляционных функций')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=150)
    plt.show()


# ================= ОСНОВНАЯ ПРОГРАММА =================

# Параметры
n = 12  # размер матриц
n_steps = 5000  # количество шагов моделирования

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА: РАНДОМИЗИРОВАННАЯ ЦЕПЬ МАРКОВА")
print("=" * 60)

# 1. Генерация двух случайных двустохастических матриц
print("\n1. ГЕНЕРАЦИЯ ДВУСТОХАСТИЧЕСКИХ МАТРИЦ")
print("-" * 40)

print("Генерируем матрицу 1...")
P1 = generate_doubly_stochastic_matrix(n)
print("Генерируем матрицу 2...")
P2 = generate_doubly_stochastic_matrix(n)

print("\nМатрица 1 (первые 5x5):")
print(np.round(P1[:5, :5], 3))
print("\nМатрица 2 (первые 5x5):")
print(np.round(P2[:5, :5], 3))

# Визуализация матриц
plot_heatmap(P1, 'Двустохастическая матрица 1', 'matrix1_heatmap')
plot_heatmap(P2, 'Двустохастическая матрица 2', 'matrix2_heatmap')

# 2. Проверка свойств двустохастичности
print("\n2. ПРОВЕРКА ДВУСТОХАСТИЧНОСТИ")
print("-" * 40)

print(f"Матрица 1 двустохастическая: {check_doubly_stochastic(P1)}")
print(f"Суммы по строкам (первые 5): {P1[:5].sum(axis=1)}")
print(f"Суммы по столбцам (первые 5): {P1[:, :5].sum(axis=0)}")

print(f"\nМатрица 2 двустохастическая: {check_doubly_stochastic(P2)}")
print(f"Суммы по строкам (первые 5): {P2[:5].sum(axis=1)}")
print(f"Суммы по столбцам (первые 5): {P2[:, :5].sum(axis=0)}")

# 3-4. Моделирование цепей Маркова
print("\n3-4. МОДЕЛИРОВАНИЕ ЦЕПЕЙ МАРКОВА")
print("-" * 40)

# Случайное начальное состояние
initial_state = np.random.randint(0, n)
print(f"Начальное состояние: {initial_state}")

print(f"Моделируем цепь 1 на {n_steps} шагов...")
trajectory1 = simulate_markov_chain(P1, initial_state, n_steps)
print(f"Моделируем цепь 2 на {n_steps} шагов...")
trajectory2 = simulate_markov_chain(P2, initial_state, n_steps)

print("Первые 20 состояний цепи 1:", trajectory1[:20])
print("Первые 20 состояний цепи 2:", trajectory2[:20])

# Визуализация траекторий
plot_trajectory(trajectory1, f'Траектория цепи Маркова 1 (начало: {initial_state})', 'trajectory1')
plot_trajectory(trajectory2, f'Траектория цепи Маркова 2 (начало: {initial_state})', 'trajectory2')

# Распределение состояний
plot_state_distribution(trajectory1, 'Распределение состояний - Цепь 1', 'distribution1')
plot_state_distribution(trajectory2, 'Распределение состояний - Цепь 2', 'distribution2')

# 5. Вычисление частот переходов
print("\n5. АНАЛИЗ ЧАСТОТ ПЕРЕХОДОВ")
print("-" * 40)

freq1 = compute_transition_frequencies(trajectory1, n)
freq2 = compute_transition_frequencies(trajectory2, n)

print("Частоты переходов для цепи 1 (первые 5x5):")
print(np.round(freq1[:5, :5], 3))
print("\nЧастоты переходов для цепи 2 (первые 5x5):")
print(np.round(freq2[:5, :5], 3))

# Сравнение с теоретическими вероятностями
plot_transition_frequencies(freq1, P1, 'Сравнение частот переходов - Матрица 1', 'freq_comparison1')
plot_transition_frequencies(freq2, P2, 'Сравнение частот переходов - Матрица 2', 'freq_comparison2')

# 6. СРАВНЕНИЕ ПОВЕДЕНИЯ ЦЕПЕЙ (отдельный рисунок)
print("\n6. СРАВНЕНИЕ ПОВЕДЕНИЯ ЦЕПЕЙ")
print("-" * 40)

# Вычисление средних и дисперсий
print(f"Цепь 1 - Среднее состояние: {np.mean(trajectory1):.3f}, Дисперсия: {np.var(trajectory1):.3f}")
print(f"Цепь 2 - Среднее состояние: {np.mean(trajectory2):.3f}, Дисперсия: {np.var(trajectory2):.3f}")
print(f"Теоретическое среднее: {(n-1)/2:.3f}, Теоретическая дисперсия: {(n**2-1)/12:.3f}")

# Построение отдельного графика для пункта 6
plot_behavior_comparison(trajectory1, trajectory2, 
                        'Рисунок 6: Сравнение поведения двух цепей Маркова', 
                        'figure6_behavior_comparison')

# 7. АВТОКОРРЕЛЯЦИОННЫЙ АНАЛИЗ (отдельный рисунок)
print("\n7. АВТОКОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("-" * 40)

max_lag = 100
autocorr1 = compute_autocorrelation(trajectory1, max_lag)
autocorr2 = compute_autocorrelation(trajectory2, max_lag)

print(f"Автокорреляция с лагом 1 - Цепь 1: {autocorr1[1]:.4f}, Цепь 2: {autocorr2[1]:.4f}")
print(f"Автокорреляция с лагом 5 - Цепь 1: {autocorr1[5]:.4f}, Цепь 2: {autocorr2[5]:.4f}")
print(f"Автокорреляция с лагом 10 - Цепь 1: {autocorr1[10]:.4f}, Цепь 2: {autocorr2[10]:.4f}")

# Построение отдельного графика для пункта 7
plot_autocorrelation_detailed(autocorr1, autocorr2, n_steps,
                             'Рисунок 7: Детальный автокорреляционный анализ', 
                             'figure7_autocorrelation_detailed')

# ================= ВЫВОДЫ =================
print("\n" + "=" * 60)
print("ВЫВОДЫ")
print("=" * 60)

print("\n1. Двустохастические матрицы успешно сгенерированы и проверены.")
print("   Обе матрицы удовлетворяют условиям: суммы по строкам и столбцам равны 1.")

print("\n2. Моделирование цепей Маркова показало:")
print(f"   - Цепь 1 посетила {len(np.unique(trajectory1))} различных состояний")
print(f"   - Цепь 2 посетила {len(np.unique(trajectory2))} различных состояний")
print(f"   - Средние значения: {np.mean(trajectory1):.2f} и {np.mean(trajectory2):.2f} "
      f"(теоретическое: {(n-1)/2:.2f})")

print("\n3. Анализ частот переходов:")
diff1 = np.mean(np.abs(freq1 - P1))
diff2 = np.mean(np.abs(freq2 - P2))
print(f"   - Среднее абсолютное отклонение для цепи 1: {diff1:.4f}")
print(f"   - Среднее абсолютное отклонение для цепи 2: {diff2:.4f}")
print(f"   - Эмпирические частоты близки к теоретическим вероятностям "
      f"(ошибка ~{np.mean([diff1, diff2]):.2%})")

print("\n4. Автокорреляционный анализ:")
print(f"   - Для цепи 1 автокорреляция затухает до нуля примерно через "
      f"{np.argmax(np.abs(autocorr1[1:]) < 0.1) if any(np.abs(autocorr1[1:]) < 0.1) else '>100'} лагов")
print(f"   - Для цепи 2 автокорреляция затухает до нуля примерно через "
      f"{np.argmax(np.abs(autocorr2[1:]) < 0.1) if any(np.abs(autocorr2[1:]) < 0.1) else '>100'} лагов")

print("\n5. Сравнение поведения двух цепей:")
if diff1 < diff2:
    print("   - Цепь 1 лучше соответствует теоретическим вероятностям")
else:
    print("   - Цепь 2 лучше соответствует теоретическим вероятностям")

if np.std(autocorr1) < np.std(autocorr2):
    print("   - Цепь 1 имеет более стабильную автокорреляционную структуру")
else:
    print("   - Цепь 2 имеет более стабильную автокорреляционную структуру")

print("\n6. Общий вывод:")
print("   Обе цепи Маркова демонстрируют ожидаемое поведение для двустохастических матриц.")
print("   Стационарное распределение близко к равномерному, автокорреляция затухает со временем.")
print("   Небольшие различия в поведении обусловлены случайной природой матриц и конечностью выборки.")

print("\n" + "=" * 60)
print("Все рисунки сохранены в директории 'figures/'")
print("=" * 60)

# Список сохраненных файлов
print("\nСохраненные файлы:")
for file in sorted(os.listdir('figures')):
    print(f"  - figures/{file}")