def calculate_network_parameters(works):
    """
    Расчёт параметров сетевого графа
    
    works: список словарей с ключами:
        - 'code': шифр работы (например, 'A-B')
        - 'duration': длительность работы
        - 'predecessors': список шифров предшествующих работ
    """

    works_dict = {work['code']: work for work in works}
    
    # Инициализация параметров
    for work in works:
        work['ph'] = 0  # раннее начало
        work['po'] = 0  # раннее окончание
        work['iih'] = 0  # позднее начало
        work['io'] = 0   # позднее окончание
        work['r'] = 0    # полный резерв
        work['rf'] = 0   # частный резерв (свободный)
 
    print("ПРЯМОЙ ПРОХОД (расчёт ранних сроков):")
    
    # Находим работы без предшественников (исходные)
    for work in works:
        if not work['predecessors']:
            work['ph'] = 0
            work['po'] = work['duration']
            print(f"{work['code']}: PH = 0, PO = {work['po']}")
    
    # Распространяем ранние сроки по графу
    changed = True
    while changed:
        changed = False
        for work in works:
            if work['predecessors'] and work['ph'] == 0 and work != works[0]:
                # Находим максимальное PO среди предшественников
                max_po = 0
                all_predecessors_found = True
                
                for pred_code in work['predecessors']:
                    pred = works_dict.get(pred_code)
                    if pred and pred['po'] > 0:
                        max_po = max(max_po, pred['po'])
                    elif pred:
                        all_predecessors_found = False
                        break
                
                if all_predecessors_found and max_po > 0:
                    work['ph'] = max_po
                    work['po'] = work['ph'] + work['duration']
                    print(f"{work['code']}: PH = max({', '.join(work['predecessors'])}) = {work['ph']}, "
                          f"PO = {work['ph']} + {work['duration']} = {work['po']}")
                    changed = True
    
    # Критическое время проекта
    t_kr = max(work['po'] for work in works)
    print(f"\nКритическое время проекта (Tкр) = {t_kr}")
    
    print("\nОБРАТНЫЙ ПРОХОД (расчёт поздних сроков):")
    
    # Строим словарь последователей для каждой работы
    followers = {work['code']: [] for work in works}
    for work in works:
        for pred_code in work['predecessors']:
            if pred_code in followers:
                followers[pred_code].append(work['code'])
    
    # Находим завершающие работы (без последователей)
    for work in works:
        if not followers[work['code']]:
            work['io'] = t_kr
            work['iih'] = work['io'] - work['duration']
            print(f"{work['code']}: IO = Tкр = {work['io']}, "
                  f"IH = {work['io']} - {work['duration']} = {work['iih']}")
    
    # Распространяем поздние сроки обратно
    changed = True
    while changed:
        changed = False
        for work in works:
            if followers[work['code']] and work['io'] == 0:
                # Находим минимальное IIH среди последователей
                min_iih = float('inf')
                all_followers_found = True
                
                for fol_code in followers[work['code']]:
                    fol = works_dict.get(fol_code)
                    if fol and fol['iih'] > 0:
                        min_iih = min(min_iih, fol['iih'])
                    elif fol:
                        all_followers_found = False
                        break
                
                if all_followers_found and min_iih != float('inf'):
                    work['io'] = min_iih
                    work['iih'] = work['io'] - work['duration']
                    print(f"{work['code']}: IO = min({', '.join(followers[work['code']])}) = {work['io']}, "
                          f"IH = {work['io']} - {work['duration']} = {work['iih']}")
                    changed = True
    
    print("\nРАСЧЁТ РЕЗЕРВОВ:")
    
    for work in works:
        # Полный резерв
        work['r'] = work['io'] - work['po']
        
        # Частный резерв (свободный)
        if followers[work['code']]:
            min_ph_fol = min(works_dict[fol]['ph'] for fol in followers[work['code']])
            work['rf'] = min_ph_fol - work['po']
        else:
            work['rf'] = None  # для завершающих работ
        
        print(f"{work['code']}: R = {work['io']} - {work['po']} = {work['r']}, "
              f"r = {work['rf'] if work['rf'] is not None else '—'}")
    
    print("\nКРИТИЧЕСКИЙ ПУТЬ:")
    
    # Находим работы с нулевым полным резервом
    critical_works = [work['code'] for work in works if work['r'] == 0]
    
    # Восстанавливаем последовательность критического пути
    critical_path = []
    current = None
    
    # Находим исходную критическую работу
    for work in works:
        if work['code'] in critical_works and not work['predecessors']:
            current = work['code']
            break
    
    # Строим путь
    while current:
        critical_path.append(current)
        # Ищем следующую критическую работу
        next_work = None
        for fol_code in followers[current]:
            if fol_code in critical_works:
                next_work = fol_code
                break
        current = next_work
    
    print(" -> ".join(critical_path))
    print(f"\nДлина критического пути: {sum(works_dict[w]['duration'] for w in critical_path)}")
    
    # ===== ВЫВОД ТАБЛИЦЫ =====
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА:")
    print("="*80)
    print(f"{'Шифр':<8} {'t':<4} {'PH':<6} {'PO':<6} {'IH':<6} {'IO':<6} {'R':<6} {'r':<6}")
    print("-"*80)
    
    for work in works:
        rf_str = f"{work['rf']:<6}" if work['rf'] is not None else f"{'—':<6}"
        print(f"{work['code']:<8} {work['duration']:<4} {work['ph']:<6} {work['po']:<6} "
              f"{work['iih']:<6} {work['io']:<6} {work['r']:<6} {rf_str}")
    
    return works, t_kr, critical_path


if __name__ == "__main__":
    
    works = [
        {'code': 'A-B', 'duration': 6, 'predecessors': []},
        {'code': 'B-C', 'duration': 5, 'predecessors': ['A-B']},
        {'code': 'A-D', 'duration': 3, 'predecessors': []},
        {'code': 'D-E', 'duration': 7, 'predecessors': ['A-D']},
        {'code': 'C-E', 'duration': 7, 'predecessors': ['B-C']},
        {'code': 'E-F', 'duration': 4, 'predecessors': ['D-E', 'C-E']}
    ]
    
    # Выполняем расчёт
    works_result, t_kr, critical_path = calculate_network_parameters(works)
    
    #print(f"\nКритическое время проекта: {t_kr}")
    #print(f"Критический путь: {' -> '.join(critical_path)}")