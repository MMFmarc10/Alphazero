# Alphazero

## 📁 Estructura del proyecto

### Raíz del proyecto

- **AlphaZero.py**  
  Clase principa de AlphaZero. Coordina las fases de self-play, entrenamiento y guardado del modelo.

- **AlphaZeroModel.py**  
  Red neuronal basada en (ResNet). Devuelve tanto la probabilidad de jugadas posibles (policy) como el valor del estado (value).

- **TestModels.py**  
  Comparar dos modelos jugando partidas usando solo las predicciones directas del modelo(sin MCTS).

- **TestModelsMultiprocessing.py**  
  Comparar dos modelos jugando partidas con MCTS. Usa múltiples procesos para ejecutar muchas partidas en paralelo.

- **Play.py**  
  Juego interactivo por consola para jugar manualmente contra un modelo entrenado.

---

## 📂 `games/`

Contiene las implementaciones de los juegos compatibles con AlphaZero.py

- **BaseGame.py**  
  Interfaz base abstracta que define los métodos que cualquier juego debe implementar para ser compatible con Alphazero.py

- **CuatroEnRaya.py**  
  Lógica completa del juego Cuatro en Raya

- **TresEnRaya.py**  
   Lógica completa del juego Tres en Raya

---

## 📂 `configs/`

Contiene clases de configuración con todos los parámetros necesarios para ejecutar AlphaZero.py


## 📂 `model_versions/`

Contiene los modelos entrenados (`.pth`) y sus configuraciones asociadas (`config.json`), organizados por ejecución de AlphaZero.py  

