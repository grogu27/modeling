"""
Лабораторная работа: Генерация графов на случайных точках
- 2 формулы вероятности: P ≈ exp(-a·d^b) и P ≈ 1/d^b
- Для каждой формулы: 3 графика без ограничений и 3 с разными ограничениями
- Всего 12 различных конфигураций
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUT_DIR = os.path.join(IMAGES_DIR, "graph_variations")

N_POINTS = 100
SIDE = 100.0


class UnionFind:
    """Система непересекающихся множеств для предотвращения циклов"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def generate_points(seed):
    """Генерация случайных точек на плоскости"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, SIDE, size=(N_POINTS, 2))


def dist_matrix(pts):
    """Матрица расстояний между точками"""
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D[i, j] = D[j, i] = d
    return D


def p_exp(d, a, b):
    """Экспоненциальная вероятность: P = exp(-a * d^b)"""
    if d <= 0:
        return 0.0
    return np.exp(-a * (d ** b))


def p_pow(d, b):
    """Степенная вероятность: P = 1 / d^b"""
    d_safe = max(d, 1e-6)
    return 1.0 / (d_safe ** b)


def build_graph(D, n_edges, prob_fn, prob_arg, rng, max_degree_cap=None, spatial_constraint=None):
    """
    Построение графа без циклов
    spatial_constraint: функция, ограничивающая возможные связи
    """
    n = D.shape[0]
    uf = UnionFind(n)
    degree = np.zeros(n, dtype=int)
    edges = []

    for _ in range(n_edges):
        # Выбор исходной вершины
        candidates_i = []
        for i in range(n):
            if max_degree_cap is not None and degree[i] >= max_degree_cap:
                continue
            # Проверяем, есть ли доступные вершины для соединения
            for j in range(n):
                if i != j and uf.find(i) != uf.find(j) and D[i, j] > 0:
                    if spatial_constraint is None or spatial_constraint(D[i, j]):
                        candidates_i.append(i)
                        break
        if not candidates_i:
            break

        # Вероятность выбора вершины пропорциональна её степени (preferential attachment)
        weights_i = np.array([degree[i] + 1 for i in candidates_i], dtype=float)
        weights_i /= weights_i.sum()
        i = rng.choice(candidates_i, p=weights_i)

        # Выбор целевой вершины
        comp_i = uf.find(i)
        candidates_j = []
        for j in range(n):
            if j != i and uf.find(j) != comp_i and D[i, j] > 0:
                if spatial_constraint is None or spatial_constraint(D[i, j]):
                    candidates_j.append(j)
        
        if not candidates_j:
            continue

        # Вероятность по формуле расстояния
        probs = np.array([prob_fn(D[i, j], prob_arg) for j in candidates_j], dtype=float)
        s = probs.sum()
        if s <= 0:
            continue
        probs /= s
        j = candidates_j[int(rng.choice(len(candidates_j), p=probs))]

        # Добавление ребра
        uf.union(i, j)
        degree[i] += 1
        degree[j] += 1
        edges.append((i, j))

    return edges, degree


def draw_graph(pts, edges, title, save_path, constraint_desc=""):
    """Визуализация графа"""
    fig, ax = plt.subplots(figsize=(10, 10))  
    
    # Рисуем точки 
    ax.scatter(pts[:, 0], pts[:, 1], s=50, c="darkblue", zorder=2, alpha=0.9, edgecolors='white', linewidth=0.8)
    
    # Рисуем рёбра 
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], 
                "gray", lw=1.2, alpha=0.7, zorder=1) 
    
    ax.set_xlim(-5, SIDE + 5)
    ax.set_ylim(-5, SIDE + 5)
    ax.set_aspect("equal")
    
    ax.set_title(f"{title}\n{constraint_desc}", fontsize=16, pad=20)
    
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    stats_text = f"Вершин: {len(pts)}\nРёбер: {len(edges)}\nСредняя степень: {2*len(edges)/len(pts):.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")  
    plt.close()


def spatial_constraint_radius(max_r):
    """Ограничение по максимальному радиусу"""
    return lambda d: d <= max_r


def spatial_constraint_annulus(r_min, r_max):
    """Ограничение в виде кольца"""
    return lambda d: r_min <= d <= r_max


def spatial_constraint_min_distance(min_r):
    """Ограничение по минимальному расстоянию"""
    return lambda d: d >= min_r


def generate_all_variations():
    """Генерация всех 12 вариаций графов"""
    
    # Конфигурации для первой формулы: P = exp(-a * d^b)
    exp_configs = [
        # Без ограничений
        {
            "id": "exp_global",
            "title": "exp(-a·d^b): Глобальные связи",
            "formula": "exp",
            "a": 0.001,
            "b": 0.5,
            "n_edges": 150,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений",
            "looks_like": "Сеть напоминает транспортную систему с хабами",
            "notes": "Малые a и b создают длинные связи, формируются узлы-хабы"
        },
        {
            "id": "exp_medium",
            "title": "exp(-a·d^b): Смешанные связи",
            "formula": "exp",
            "a": 0.05,
            "b": 1.2,
            "n_edges": 120,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений",
            "looks_like": "Баланс локальных и глобальных связей",
            "notes": "Средние параметры дают равномерное распределение связей"
        },
        {
            "id": "exp_local",
            "title": "exp(-a·d^b): Локальные связи",
            "formula": "exp",
            "a": 0.5,
            "b": 2.5,
            "n_edges": 90,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений",
            "looks_like": "Похоже на сеть ближайших соседей",
            "notes": "Большие a и b сильно подавляют дальние связи"
        },
        
        # С ограничениями
        {
            "id": "exp_radius_30",
            "title": "exp(-a·d^b): Радиус 30",
            "formula": "exp",
            "a": 0.01,
            "b": 1.0,
            "n_edges": 100,
            "max_degree_cap": 8,
            "spatial_constraint": spatial_constraint_radius(30),
            "constraint_desc": "Максимальная длина ребра: 30",
            "looks_like": "Кластерная структура с локальными группами",
            "notes": "Ограничение радиуса создаёт изолированные кластеры"
        },
        {
            "id": "exp_annulus_20_50",
            "title": "exp(-a·d^b): Кольцо 20-50",
            "formula": "exp",
            "a": 0.02,
            "b": 1.5,
            "n_edges": 80,
            "max_degree_cap": 6,
            "spatial_constraint": spatial_constraint_annulus(20, 50),
            "constraint_desc": "Длина ребра в интервале [20, 50]",
            "looks_like": "Сеть с 'запрещённой зоной' вокруг узлов",
            "notes": "Кольцевое ограничение создаёт интересную топологию без коротких связей"
        },
        {
            "id": "exp_min_dist_15",
            "title": "exp(-a·d^b): Мин. расстояние 15",
            "formula": "exp",
            "a": 0.03,
            "b": 1.8,
            "n_edges": 70,
            "max_degree_cap": 5,
            "spatial_constraint": spatial_constraint_min_distance(15),
            "constraint_desc": "Минимальная длина ребра: 15",
            "looks_like": "Разреженная сеть дальних связей",
            "notes": "Отсутствие коротких связей приводит к формированию 'дальнобойных' соединений"
        },
    ]
    
    # Конфигурации для второй формулы: P = 1 / d^b
    pow_configs = [
        # Без ограничений
        {
            "id": "pow_global",
            "title": "1/d^b: Глобальные связи (малый b)",
            "formula": "pow",
            "b": 0.3,
            "n_edges": 140,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений, b=0.3",
            "looks_like": "Почти полный граф с предпочтением дальних связей",
            "notes": "Очень малый b делает вероятность почти не зависящей от расстояния"
        },
        {
            "id": "pow_medium",
            "title": "1/d^b: Сбалансированные связи",
            "formula": "pow",
            "b": 1.0,
            "n_edges": 120,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений, b=1.0",
            "looks_like": "Естественная пространственная сеть",
            "notes": "b=1 даёт классическое гравитационное притяжение"
        },
        {
            "id": "pow_local",
            "title": "1/d^b: Локальные связи (большой b)",
            "formula": "pow",
            "b": 2.5,
            "n_edges": 90,
            "max_degree_cap": None,
            "spatial_constraint": None,
            "constraint_desc": "Без пространственных ограничений, b=2.5",
            "looks_like": "Сильно локальная сеть с кластеризацией",
            "notes": "Большой b резко подавляет дальние связи"
        },
        
        # С ограничениями
        {
            "id": "pow_radius_25",
            "title": "1/d^b: Радиус 25, b=1.5",
            "formula": "pow",
            "b": 1.5,
            "n_edges": 100,
            "max_degree_cap": 7,
            "spatial_constraint": spatial_constraint_radius(25),
            "constraint_desc": "Макс. радиус: 25, ограничение степени: 7",
            "looks_like": "Локальные кластеры с ограниченной связностью",
            "notes": "Комбинация радиуса и ограничения степени создаёт структуру, похожую на сенсорные сети"
        },
        {
            "id": "pow_annulus_10_40",
            "title": "1/d^b: Кольцо 10-40, b=2.0",
            "formula": "pow",
            "b": 2.0,
            "n_edges": 85,
            "max_degree_cap": 6,
            "spatial_constraint": spatial_constraint_annulus(10, 40),
            "constraint_desc": "Длина ребра [10,40], макс. степень: 6",
            "looks_like": "Сеть с равномерным распределением длин связей",
            "notes": "Кольцевое ограничение создаёт равномерную mesh-сеть"
        },
        {
            "id": "pow_min_dist_20",
            "title": "1/d^b: Мин. расстояние 20, b=1.2",
            "formula": "pow",
            "b": 1.2,
            "n_edges": 75,
            "max_degree_cap": 5,
            "spatial_constraint": spatial_constraint_min_distance(20),
            "constraint_desc": "Мин. длина ребра: 20, макс. степень: 5",
            "looks_like": "Разреженная сеть с длинными мостами",
            "notes": "Напоминает магистральные сети связи между удалёнными узлами"
        },
    ]
    
    # Объединяем все конфигурации
    all_configs = exp_configs + pow_configs
    
    # Создаём выходную директорию
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Отчёт о результатах
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ОТЧЁТ ПО ГЕНЕРАЦИИ ГРАФОВ")
    report_lines.append("=" * 80 + "\n")
    
    # Генерируем графы для каждой конфигурации
    for idx, cfg in enumerate(all_configs, 1):
        # Создаём поддиректорию для конфигурации
        subdir = os.path.join(OUT_DIR, f"{idx:02d}_{cfg['id']}")
        os.makedirs(subdir, exist_ok=True)
        
        cfg_report = []
        cfg_report.append(f"\nКонфигурация {idx}: {cfg['title']}")
        cfg_report.append("-" * 60)
        cfg_report.append(f"Формула: {cfg['formula']}")
        if cfg['formula'] == 'exp':
            cfg_report.append(f"Параметры: a={cfg['a']}, b={cfg['b']}")
        else:
            cfg_report.append(f"Параметр: b={cfg['b']}")
        cfg_report.append(f"Ограничения: {cfg['constraint_desc']}")
        cfg_report.append(f"Целевое число рёбер: {cfg['n_edges']}")
        cfg_report.append(f"Визуально похоже на: {cfg['looks_like']}")
        cfg_report.append(f"Наблюдения: {cfg['notes']}")
        cfg_report.append("")
        
        # Генерируем 3 графика для каждой конфигурации
        graphs_data = []
        
        for k in range(1, 4):
            # Разные seed для разнообразия
            seed_pts = 5000 + idx * 50 + k * 100
            seed_rng = 6000 + idx * 70 + k * 150
            
            pts = generate_points(seed=seed_pts)
            D = dist_matrix(pts)
            rng = np.random.default_rng(seed_rng)
            
            # Настройка функции вероятности
            if cfg["formula"] == "exp":
                a, b = cfg["a"], cfg["b"]
                prob_fn = lambda d, _: p_exp(d, a, b)
                prob_arg = None
                formula_str = f"P=exp(-{a}·d^{b})"
            else:
                b = cfg["b"]
                prob_fn = lambda d, x: p_pow(d, x)
                prob_arg = b
                formula_str = f"P=1/d^{b}"
            
            # Построение графа
            edges, degree = build_graph(
                D, 
                cfg["n_edges"], 
                prob_fn, 
                prob_arg, 
                rng, 
                cfg["max_degree_cap"],
                cfg["spatial_constraint"]
            )
            
            graphs_data.append({
                'edges': edges,
                'degree': degree,
                'seed_pts': seed_pts,
                'seed_rng': seed_rng
            })
            
            # Сохранение изображения
            title = f"{idx}. {formula_str}"
            filename = f"graph_{k}.png"
            save_path = os.path.join(subdir, filename)
            draw_graph(pts, edges, title, save_path, cfg['constraint_desc'])
        
        # Анализ для этой конфигурации
        avg_edges = np.mean([len(g['edges']) for g in graphs_data])
        avg_degree = np.mean([2*len(g['edges'])/N_POINTS for g in graphs_data])
        
        cfg_report.append(f"Результаты (среднее по 3 графикам):")
        cfg_report.append(f"  - Фактическое число рёбер: {avg_edges:.1f}")
        cfg_report.append(f"  - Средняя степень вершины: {avg_degree:.2f}")
        cfg_report.append("")
        
        report_lines.extend(cfg_report)
        
        print(f"Сгенерирована конфигурация {idx}: {cfg['id']}")
    
    # Сохраняем отчёт
    report_path = os.path.join(OUT_DIR, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nГенерация завершена! Результаты в: {OUT_DIR}")
    print(f"Отчёт сохранён в: {report_path}")


if __name__ == "__main__":
    generate_all_variations()