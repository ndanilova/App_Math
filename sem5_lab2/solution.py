import numpy as np
import json
from scipy.optimize import linprog

with open('tasks/task2solution.py.json', 'r') as file:
    data = json.load(file)
    game_matrix = np.array(data['matrix'])

row_min = np.min(game_matrix, axis=1)
col_max = np.max(game_matrix, axis=0)
simplexed_matrix = np.where(game_matrix == row_min[:, np.newaxis], row_min[:, np.newaxis], game_matrix)
simplexed_matrix = np.where(simplexed_matrix == col_max, col_max, simplexed_matrix)

row_maxs = np.max(simplexed_matrix, axis=1)
col_mins = np.min(simplexed_matrix, axis=0)
optimal_value = np.max(col_mins)

if optimal_value in row_maxs:
    optimal_row_strategy = np.argmax(row_maxs)
    optimal_col_strategy = np.where(row_maxs == optimal_value)[0][0]
    print("Оптимальные стратегии:", optimal_row_strategy, optimal_col_strategy)
    print("Цена игры:", optimal_value)
else:
    num_row_strategies = game_matrix.shape[0]
    num_col_strategies = game_matrix.shape[1]

    c = np.ones(num_row_strategies + num_col_strategies)  # Целевая функция
    A = -np.concatenate((game_matrix.T, np.eye(num_col_strategies)), axis=1)
    b = -np.concatenate((np.ones(num_row_strategies), np.zeros(num_col_strategies)))

    res = linprog(c, A_ub=A, b_ub=b)
    mixed_row_strategy = res.x[:num_row_strategies]
    mixed_col_strategy = res.x[num_row_strategies:]

    optimal_value = 1 / np.sum(mixed_row_strategy)

    print("Смешанные стратегии:")
    print("Стратегии игрока 1:", mixed_row_strategy)
    print("Стратегии игрока 2:", mixed_col_strategy)
    print("Цена игры:", optimal_value)