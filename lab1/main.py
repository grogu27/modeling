import numpy as np
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

f_pdf_vec = np.vectorize(f_pdf)

# Часть 2: Функция распределения (CDF)
def F_cdf(x):
    if x <= -0.2:
        return 0.0
    elif x <= 1.0:
        return (a / 4) * (x + 0.2)**4  # a/4 = 625/(4*1134)=625/4536
    elif x < 2.5:
        # Используем упрощенную формулу
        return (2/63) * (-10*x**2 + 50*x - 31)
    else:
        return 1.0

F_cdf_vec = np.vectorize(F_cdf)

# Часть 3: Обратная функция (квантильная)
def F_inv(y):
    if y < 0.0:
        return -0.2
    elif y <= F1:
        # Первый участок
        return ((4536 / 625) * y)**(1/4) - 0.2
    elif y <= 1.0:
        # Второй участок
        return (25 - 3 * np.sqrt(35 * (1 - y))) / 10
    else:
        return 2.5

F_inv_vec = np.vectorize(F_inv)

# Часть 4: Генератор случайных чисел
def generate_samples(n_samples):
    """Генерирует выборку методом обратного преобразования"""
    u = np.random.uniform(0, 1, n_samples)
    samples = F_inv_vec(u)
    return samples

# --- Создание отдельных графиков и сохранение в файлы ---

# 1. График плотности вероятности
plt.figure(figsize=(10, 8))
x_vals = np.linspace(-0.5, 3.0, 500)
y_vals = f_pdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='x=1')
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='x=2.5')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.fill_between(x_vals, y_vals, where=[(-0.2 <= x <= 2.5) for x in x_vals], alpha=0.2, color='blue')
plt.title('Функция плотности вероятности f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_pdf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_pdf.png")

# 2. График функции распределения
plt.figure(figsize=(10, 8))
x_vals = np.linspace(-0.5, 3.0, 500)
y_vals = F_cdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='F(x)')
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=F1, color='orange', linestyle=':', alpha=0.7, label=f'y = {F1:.3f} (F(1))')
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
plt.title('Функция распределения F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_cdf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_cdf.png")

# 3. График обратной функции (квантили)
plt.figure(figsize=(10, 8))
y_vals = np.linspace(0, 1, 500)
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
samples = generate_samples(10000)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Выборка (n=10000)')
x_vals = np.linspace(-0.2, 2.5, 300)
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

# 5. Сводный график (все 4 в одном)
plt.figure(figsize=(14, 10))

# 5.1 График плотности вероятности
plt.subplot(2, 2, 1)
x_vals = np.linspace(-0.5, 3.0, 500)
y_vals = f_pdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='x=1')
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='x=2.5')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.fill_between(x_vals, y_vals, where=[(-0.2 <= x <= 2.5) for x in x_vals], alpha=0.2, color='blue')
plt.title('Функция плотности вероятности f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 График функции распределения
plt.subplot(2, 2, 2)
x_vals = np.linspace(-0.5, 3.0, 500)
y_vals = F_cdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='F(x)')
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=F1, color='orange', linestyle=':', alpha=0.7, label=f'y = {F1:.3f} (F(1))')
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
plt.title('Функция распределения F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 График обратной функции
plt.subplot(2, 2, 3)
y_vals = np.linspace(0, 1, 500)
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

# 5.4 Гистограмма выборки
plt.subplot(2, 2, 4)
samples = generate_samples(10000)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Выборка (n=10000)')
x_vals = np.linspace(-0.2, 2.5, 300)
y_vals = f_pdf_vec(x_vals)
plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Теоретическая f(x)')
plt.title('Гистограмма выборки и теоретическая плотность')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_all.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранен файл: plot_all.png")

# Вывод информации в консоль
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ РАСЧЕТОВ")
print("="*50)
print(f"Параметр a = {a:.6f} (дробь: {a})")
print(f"Параметр b = {b:.6f} (дробь: {b})")
print(f"F(1) = {F1:.6f} (дробь: {F1})")
print(f"\nПервые 20 сгенерированных значений:")
for i, val in enumerate(samples[:20], 1):
    print(f"{i:2d}: {val:.6f}")

print(f"\nСтатистика выборки (n=10000):")
print(f"Минимум: {np.min(samples):.4f}")
print(f"Максимум: {np.max(samples):.4f}")
print(f"Среднее: {np.mean(samples):.4f}")
print(f"Стандартное отклонение: {np.std(samples):.4f}")

print("\nВсе графики сохранены в текущую директорию:")
print("  - plot_pdf.png (плотность вероятности)")
print("  - plot_cdf.png (функция распределения)")
print("  - plot_inv.png (обратная функция)")
print("  - plot_histogram.png (гистограмма)")
print("  - plot_all.png (все графики вместе)")