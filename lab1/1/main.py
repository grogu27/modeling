import math
import random
import matplotlib.pyplot as plt

# Параметры
a = 625 / 1134
b = 40 / 63
F1 = 2 / 7  # F(1)

# Часть 1: Функция плотности вероятности (PDF)
def f_pdf(x):
    if x < -0.2:
        return 0.0
    elif x <= 1.0:
        return a * (x + 0.2)**3
    elif x < 2.5:
        return b * (2.5 - x)
    else:
        return 0.0

# Часть 2: Функция распределения (CDF)
def F_cdf(x):
    if x <= -0.2:
        return 0.0
    elif x <= 1.0:
        return (a / 4) * (x + 0.2)**4
    elif x < 2.5:
        return (2/63) * (-10*x**2 + 50*x - 31)
    else:
        return 1.0

# Часть 3: Обратная функция (квантильная)
def F_inv(y):
    if y < 0.0:
        return -0.2
    elif y <= F1:
        # Первый участок: корень 4-й степени
        return ((4536 / 625) * y)**(1/4) - 0.2
    elif y <= 1.0:
        # Второй участок
        return (25 - 3 * math.sqrt(35 * (1 - y))) / 10
    else:
        return 2.5

# Часть 4: Генератор случайных чисел
def generate_samples(n_samples):
    """Генерирует выборку методом обратного преобразования"""
    samples = []
    for _ in range(n_samples):
        u = random.random()  # равномерное число от 0 до 1
        samples.append(F_inv(u))
    return samples

# Вспомогательные функции для создания массивов для графиков
def linspace(start, stop, num):
    """Создает массив из num точек между start и stop"""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

def vectorize_func(func):
    """Создает векторизованную версию функции"""
    def vectorized(x_list):
        if isinstance(x_list, (list, tuple)):
            return [func(x) for x in x_list]
        else:
            return func(x_list)
    return vectorized

# Создаем векторизованные версии функций для графиков
f_pdf_vec = vectorize_func(f_pdf)
F_cdf_vec = vectorize_func(F_cdf)
F_inv_vec = vectorize_func(F_inv)

# Генерируем выборку для всех графиков
n_samples = 20000
samples = generate_samples(n_samples)
print(f"Сгенерировано {n_samples} значений")

# --- Создание отдельных графиков и сохранение в файлы ---

# 1. График плотности вероятности с облаком точек
plt.figure(figsize=(10, 8))
x_vals = linspace(-0.5, 3.0, 500)
y_vals = f_pdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Теоретическая f(x)')

# Облако точек для визуализации плотности (берем 3000 точек для наглядности)
n_scatter = min(3000, len(samples))
scatter_indices = random.sample(range(len(samples)), n_scatter)
scatter_x = [samples[i] for i in scatter_indices]
scatter_y = [random.uniform(0, f_pdf(x)) for x in scatter_x]
plt.scatter(scatter_x, scatter_y, color='darkblue', s=0.5, alpha=0.3, 
            label='Точки датчика (облако)', rasterized=True)

plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='x=1')
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='x=2.5')
plt.axhline(y=0, color='black', linewidth=0.5)

# Заливка под кривой
fill_x = [x for x in x_vals if -0.2 <= x <= 2.5]
fill_y = [f_pdf(x) for x in fill_x]
plt.fill_between(fill_x, fill_y, alpha=0.1, color='blue')

plt.title('Функция плотности вероятности f(x) с визуализацией работы генератора')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('plot_pdf_with_cloud.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_pdf_with_cloud.png")

# 2. График функции распределения с эмпирической CDF
plt.figure(figsize=(10, 8))
x_vals = linspace(-0.5, 3.0, 500)
y_vals = F_cdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Теоретическая F(x)')

# Эмпирическая функция распределения
sorted_samples = sorted(samples)
y_ecdf = [i / len(sorted_samples) for i in range(len(sorted_samples))]
plt.step(sorted_samples, y_ecdf, where='post', color='blue', alpha=0.6, 
         linewidth=1.5, label='Эмпирическая F(x)')

plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=F1, color='orange', linestyle=':', alpha=0.7, label=f'y = {F1:.3f} (F(1))')
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
plt.title('Функция распределения F(x) с эмпирической CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_cdf_with_empirical.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_cdf_with_empirical.png")

# 3. График обратной функции (квантили)
plt.figure(figsize=(10, 8))
y_vals = linspace(0, 1, 500)
x_vals = F_inv_vec(y_vals)
plt.plot(y_vals, x_vals, 'g-', linewidth=2, label='F^{-1}(y)')
plt.axvline(x=F1, color='orange', linestyle=':', alpha=0.7, label=f'y = {F1:.3f}')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=2.5, color='gray', linestyle='--', alpha=0.7)
plt.title('Обратная функция (квантильная)')
plt.xlabel('y')
plt.ylabel('x = F^{-1}(y)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_inv.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_inv.png")

# 4. Гистограмма выборки
plt.figure(figsize=(10, 8))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label='Выборка (n={})'.format(n_samples))
x_vals = linspace(-0.2, 2.5, 300)
y_vals = f_pdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Теоретическая f(x)')
plt.title('Гистограмма выборки и теоретическая плотность')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_histogram.png")

# 5. Сводный график с улучшениями
fig = plt.figure(figsize=(16, 12))

# 5.1 График плотности вероятности с облаком точек
ax1 = plt.subplot(2, 2, 1)
x_vals = linspace(-0.5, 3.0, 500)
y_vals = f_pdf_vec(x_vals)
ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='Теоретическая f(x)')

# Облако точек
n_scatter = min(2000, len(samples))
scatter_indices = random.sample(range(len(samples)), n_scatter)
scatter_x = [samples[i] for i in scatter_indices]
scatter_y = [random.uniform(0, f_pdf(x)) for x in scatter_x]
ax1.scatter(scatter_x, scatter_y, color='darkblue', s=0.3, alpha=0.2, 
            label='Точки датчика', rasterized=True)

ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='x=1')
ax1.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='x=2.5')
ax1.axhline(y=0, color='black', linewidth=0.5)
fill_x = [x for x in x_vals if -0.2 <= x <= 2.5]
fill_y = [f_pdf(x) for x in fill_x]
ax1.fill_between(fill_x, fill_y, alpha=0.1, color='blue')
ax1.set_title('Плотность вероятности с облаком точек')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# 5.2 График функции распределения с эмпирической CDF
ax2 = plt.subplot(2, 2, 2)
x_vals = linspace(-0.5, 3.0, 500)
y_vals = F_cdf_vec(x_vals)
ax2.plot(x_vals, y_vals, 'r-', linewidth=2, label='Теоретическая F(x)')

# Эмпирическая CDF
sorted_samples = sorted(samples)
y_ecdf = [i / len(sorted_samples) for i in range(len(sorted_samples))]
ax2.step(sorted_samples, y_ecdf, where='post', color='blue', alpha=0.6, 
         linewidth=1, label='Эмпирическая F(x)')

ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
ax2.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
ax2.axhline(y=F1, color='orange', linestyle=':', alpha=0.7, label=f'F(1)={F1:.3f}')
ax2.axhline(y=1.0, color='black', linewidth=0.5)
ax2.set_title('Функция распределения')
ax2.set_xlabel('x')
ax2.set_ylabel('F(x)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 5.3 График обратной функции
ax3 = plt.subplot(2, 2, 3)
y_vals = linspace(0, 1, 500)
x_vals = F_inv_vec(y_vals)
ax3.plot(y_vals, x_vals, 'g-', linewidth=2, label='F^{-1}(y)')
ax3.axvline(x=F1, color='orange', linestyle=':', alpha=0.7, label=f'y={F1:.3f}')
ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax3.axhline(y=2.5, color='gray', linestyle='--', alpha=0.7)
ax3.set_title('Обратная функция (квантильная)')
ax3.set_xlabel('y')
ax3.set_ylabel('x = F^{-1}(y)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 5.4 Гистограмма выборки
ax4 = plt.subplot(2, 2, 4)
ax4.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label='Выборка')
x_vals = linspace(-0.2, 2.5, 300)
y_vals = f_pdf_vec(x_vals)
ax4.plot(x_vals, y_vals, 'r-', linewidth=2, label='Теоретическая f(x)')
ax4.set_title('Гистограмма выборки')
ax4.set_xlabel('x')
ax4.set_ylabel('Плотность')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_all_enhanced.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_all_enhanced.png")

# Статистика для выборки
def mean(arr):
    return sum(arr) / len(arr)

def variance(arr):
    m = mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)

def std_dev(arr):
    return math.sqrt(variance(arr))

def min_val(arr):
    return min(arr)

def max_val(arr):
    return max(arr)

# Вывод информации в консоль
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ РАСЧЕТОВ")
print("="*60)
print(f"Параметр a = {a:.6f} (дробь: {a})")
print(f"Параметр b = {b:.6f} (дробь: {b})")
print(f"F(1) = {F1:.6f} (дробь: {F1})")
print(f"\nПервые 20 сгенерированных значений:")
for i, val in enumerate(samples[:20], 1):
    print(f"{i:2d}: {val:.6f}")

print(f"\nСтатистика выборки (n={n_samples}):")
print(f"Минимум: {min_val(samples):.4f}")
print(f"Максимум: {max_val(samples):.4f}")
print(f"Среднее: {mean(samples):.4f}")
print(f"Стандартное отклонение: {std_dev(samples):.4f}")
print(f"Медиана: {sorted_samples[len(sorted_samples)//2]:.4f}")

# Проверка на соответствие F(1)
ecdf_at_1 = sum(1 for x in samples if x <= 1.0) / len(samples)
print(f"\nЭмпирическая F(1): {ecdf_at_1:.4f} (теоретическая: {F1:.4f})")
print(f"Разница: {abs(ecdf_at_1 - F1):.4f}")

print("\nВсе графики сохранены в текущую директорию:")
print("  - plot_pdf_with_cloud.png (плотность + облако точек)")
print("  - plot_cdf_with_empirical.png (CDF + эмпирическая CDF)")
print("  - plot_inv.png (обратная функция)")
print("  - plot_histogram.png (гистограмма)")
print("  - plot_all_enhanced.png (все графики вместе с улучшениями)")