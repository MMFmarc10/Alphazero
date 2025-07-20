wins1 = 105
wins2 = 95
draws =  0

# Total de partidas jugadas
total_games = wins1 + wins2 + draws

# Score del modelo actual (modelo1): victoria = 1, empate = 0.5, derrota = 0
score_actual = (wins1 + 0.5 * draws) / total_games if total_games > 0 else 0

# Score del modelo best (modelo2), opcional
score_best = (wins2 + 0.5 * draws) / total_games if total_games > 0 else 0

# Mostrar resultados
print(f"\n--- Score ---")
print(f"Modelo Actual: {score_actual * 100:.2f}%")
print(f"Modelo Best:   {score_best * 100:.2f}%")