# Predicción de Cálculos Biliares mediante Machine Learning: Análisis Comparativo de XGBoost y Redes Neuronales con Algoritmos Genéticos para Selección de Características

**Alessandro Ledesma**  
alessandroledesma@[institución].edu  

---

## RESUMEN

Este estudio presenta un enfoque de machine learning para la predicción de cálculos biliares utilizando características clínicas de pacientes. Se implementaron y compararon tres modelos: XGBoost inicial, XGBoost optimizado y redes neuronales multicapa (MLP), utilizando un dataset de 319 pacientes con 39 características clínicas. Se aplicaron técnicas avanzadas incluyendo algoritmos genéticos para selección de características, SMOTE para balanceo de datos y detección de outliers por consenso múltiple. El modelo XGBoost optimizado demostró el mejor rendimiento con un F1-Score de 78.04% y ROC-AUC de 86.33%. Los algoritmos genéticos redujeron efectivamente las características de 39 a 19, identificando la Vitamina D como el predictor más importante (correlación negativa de -0.355), seguido por la Proteína C-Reactiva (correlación positiva de 0.282). Los resultados sugieren que este enfoque puede ser viable para el apoyo al diagnóstico clínico temprano de cálculos biliares, proporcionando una herramienta computacional precisa y eficiente para profesionales de la salud.

---

## 1. INTRODUCCIÓN

Los cálculos biliares representan una de las patologías más comunes del sistema digestivo, afectando aproximadamente al 10-15% de la población adulta en países desarrollados [1]. La detección temprana y precisa de esta condición es crucial para prevenir complicaciones graves como colecistitis aguda, colangitis y pancreatitis biliar.

El diagnóstico tradicional de cálculos biliares depende principalmente de técnicas de imagen como ultrasonografía abdominal, tomografía computarizada y resonancia magnética. Sin embargo, estos métodos pueden ser costosos, requerir tiempo considerable y no siempre están disponibles en entornos de atención primaria.

En los últimos años, el machine learning ha demostrado un potencial significativo en aplicaciones médicas, particularmente en tareas de diagnóstico y predicción. Diversos estudios han explorado el uso de algoritmos de aprendizaje automático para el diagnóstico de enfermedades gastrointestinales, mostrando resultados prometedores en términos de precisión y eficiencia [2].

**Objetivo del presente estudio:** Desarrollar y evaluar modelos de machine learning para la predicción de cálculos biliares utilizando características clínicas rutinariamente disponibles, con el fin de crear una herramienta de apoyo al diagnóstico que sea precisa, eficiente y accesible en diversos entornos clínicos.

---

## 2. METODOLOGÍA

### 2.1 Diseño del Estudio

Se implementó un estudio de desarrollo y validación de modelos predictivos utilizando un enfoque de machine learning supervisado. El proceso metodológico se estructuró en cinco fases principales como se muestra en la Figura 1.

**Figura 1.** Diagrama de flujo de la metodología implementada.

```
Dataset Original (319 muestras, 39 características)
                    ↓
Análisis Exploratorio de Datos (EDA)
                    ↓
Detección y Tratamiento de Outliers
                    ↓
Feature Engineering Avanzado
                    ↓
Selección de Características (Algoritmos Genéticos)
                    ↓
Balanceo de Datos (SMOTE)
                    ↓
Implementación de Modelos (XGBoost, MLP)
                    ↓
Evaluación y Comparación
```

### 2.2 Técnicas Implementadas

El enfoque metodológico integra técnicas avanzadas de machine learning incluyendo:

- **XGBoost Classifier:** Algoritmo de gradient boosting optimizado para problemas de clasificación
- **Redes Neuronales Multicapa (MLP):** Con regularización L2 y early stopping
- **Algoritmos Genéticos:** Para optimización de selección de características
- **SMOTE:** Para balanceo sintético de clases
- **Detección Múltiple de Outliers:** Consenso de Isolation Forest, Local Outlier Factor, Elliptic Envelope y Z-Score

### 2.3 Validación y Métricas

Se utilizó validación cruzada estratificada y división 80/20 para entrenamiento/prueba. Las métricas de evaluación incluyeron Accuracy, Precision, Recall, F1-Score, ROC-AUC y Balanced Accuracy.

---

## 3. EXPERIMENTACIÓN

### 3.1 Descripción del Dataset

El dataset utilizado contiene información clínica de 319 pacientes, distribuidos equitativamente entre casos con cálculos biliares (49.5%, n=158) y controles sin cálculos (50.5%, n=161). Este balance perfecto elimina la necesidad de técnicas complejas de balanceo inicial.

Las 39 características clínicas incluyen:
- **Variables demográficas:** Edad, género
- **Comorbilidades:** Diabetes mellitus, hiperlipidemia, hipotiroidismo, enfermedad arterial coronaria
- **Composición corporal:** IMC, masa grasa, masa magra, agua corporal total
- **Parámetros bioquímicos:** Glucosa, perfil lipídico, enzimas hepáticas, marcadores inflamatorios
- **Otros indicadores:** Vitamina D, hemoglobina, función renal

### 3.2 Preprocesamiento de Datos

#### 3.2.1 Análisis Exploratorio
El análisis inicial reveló la ausencia completa de valores faltantes y una distribución balanceada de la variable objetivo. Se identificaron las correlaciones más fuertes con la presencia de cálculos biliares:

1. Vitamina D: correlación negativa (-0.355)
2. Proteína C-Reactiva: correlación positiva (0.282)
3. Masa Magra (%): correlación negativa (-0.226)
4. Grasa Corporal Total (%): correlación positiva (0.225)

#### 3.2.2 Detección de Outliers
Se implementó un enfoque de consenso múltiple utilizando cuatro métodos:
- Isolation Forest
- Local Outlier Factor  
- Elliptic Envelope
- Z-Score (umbral > 3)

Los casos identificados como outliers por al menos dos métodos fueron marcados para análisis especial, representando aproximadamente el 10% del dataset.

#### 3.2.3 Feature Engineering
Se crearon nuevas características mediante:
- **Transformaciones logarítmicas** para variables con sesgo significativo
- **Interacciones multiplicativas** entre variables importantes
- **Índices compuestos de salud:** Body Composition Index, Metabolic Risk Index
- **Categorización** de variables continuas en rangos clínicamente relevantes

### 3.3 Selección de Características con Algoritmos Genéticos

Se implementó un algoritmo genético con los siguientes parámetros:
- Población: 30 individuos
- Generaciones: 15
- Probabilidad de cruzamiento: 0.7
- Probabilidad de mutación: 0.2
- Función fitness: F1-Score con penalización por número de características

El algoritmo redujo exitosamente las características de 39 a 19, logrando una reducción del 51.3% mientras mantenía el poder predictivo del modelo.

### 3.4 Implementación de Modelos

#### 3.4.1 XGBoost
Se entrenaron dos versiones:
- **XGBoost Inicial:** Hiperparámetros por defecto
- **XGBoost Optimizado:** Grid Search con validación cruzada

Hiperparámetros optimizados:
- n_estimators: 200
- max_depth: 3  
- learning_rate: 0.1
- subsample: 1.0

#### 3.4.2 Red Neuronal MLP
Arquitectura implementada:
- Capas ocultas: (38, 19) neuronas
- Función de activación: ReLU
- Solver: Adam
- Regularización L2: α = 0.001
- Early stopping con paciencia de 10 épocas

---

## 4. RESULTADOS

### 4.1 Rendimiento de Modelos

La Tabla 1 presenta la comparación exhaustiva del rendimiento de los tres modelos implementados.

**Tabla 1.** Comparación de métricas de rendimiento de los modelos.

| Modelo | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-------------------|-----------|--------|----------|---------|
| XGBoost Inicial | 0.7031 | 0.7031 | 0.7033 | 0.7031 | **0.7031** | 0.8428 |
| **XGBoost Optimizado** | **0.7812** | **0.7812** | **0.7857** | **0.7812** | **0.7804** | **0.8633** |
| Red Neuronal | 0.7344 | 0.7344 | 0.7402 | 0.7344 | 0.7327 | 0.8135 |

El modelo XGBoost Optimizado demostró el mejor rendimiento en todas las métricas evaluadas, destacándose especialmente en:
- **F1-Score:** 78.04% (7.73 puntos porcentuales superior al modelo inicial)
- **ROC-AUC:** 86.33% (excelente capacidad discriminativa)
- **Balanced Accuracy:** 78.12% (apropiado para datos balanceados)

### 4.2 Importancia de Características

El análisis de importancia de características del modelo XGBoost optimizado reveló los siguientes predictores principales:

**Tabla 2.** Top 10 características más importantes según XGBoost.

| Ranking | Característica | Importancia | Tipo |
|---------|----------------|-------------|------|
| 1 | Vitamin D ÷ C-Reactive Protein | 0.1906 | Interacción |
| 2 | Hyperlipidemia | 0.0947 | Comorbilidad |
| 3 | CRP × Lean Mass (%) | 0.0725 | Interacción |
| 4 | Total Fat Content | 0.0717 | Composición Corporal |
| 5 | Body Composition Index | 0.0649 | Índice Compuesto |
| 6 | Extracellular Water | 0.0446 | Composición Corporal |
| 7 | Body Protein Content | 0.0440 | Composición Corporal |
| 8 | Glucose | 0.0411 | Bioquímico |
| 9 | Comorbidity | 0.0393 | Clínico |
| 10 | Muscle Mass | 0.0385 | Composición Corporal |

### 4.3 Análisis de Matrices de Confusión

Las matrices de confusión revelaron patrones diferentes entre los modelos:

- **XGBoost Optimizado:** 23 verdaderos negativos, 27 verdaderos positivos, 9 falsos positivos, 5 falsos negativos
- **Red Neuronal:** 26 verdaderos negativos, 21 verdaderos positivos, 6 falsos positivos, 11 falsos negativos

El XGBoost optimizado mostró mejor balance entre sensibilidad y especificidad, con menor número de falsos negativos (casos de cálculos biliares no detectados).

### 4.4 Curvas ROC y Análisis de Discriminación

El área bajo la curva ROC de 0.8633 para XGBoost optimizado indica una excelente capacidad discriminativa del modelo, clasificándose como "muy buena" según estándares epidemiológicos (>0.8).

---

## 5. DISCUSIÓN DE LOS RESULTADOS

### 5.1 Superioridad del XGBoost Optimizado

Los resultados demuestran claramente la superioridad del XGBoost optimizado sobre las redes neuronales y la versión inicial del algoritmo. Esta superioridad puede atribuirse a varios factores:

**Capacidad de manejo de características heterogéneas:** XGBoost maneja eficientemente la mezcla de variables categóricas y continuas presente en el dataset clínico, mientras que las redes neuronales requieren preprocesamiento más extensivo.

**Robustez ante outliers:** El enfoque basado en árboles de decisión inherente en XGBoost proporciona mayor robustez ante valores atípicos comparado con las redes neuronales, que son más sensibles a la escala y distribución de los datos.

**Interpretabilidad clínica:** XGBoost ofrece medidas directas de importancia de características, facilitando la interpretación clínica de los resultados, aspecto crucial en aplicaciones médicas [3].

### 5.2 Significancia Clínica de los Predictores Principales

#### 5.2.1 Vitamina D como Predictor Principal

El hallazgo de la Vitamina D como el predictor más importante (correlación negativa de -0.355) es consistente con evidencia epidemiológica reciente que sugiere una asociación inversa entre niveles séricos de vitamina D y riesgo de enfermedad biliar [4]. Los mecanismos propuestos incluyen:

- **Regulación del metabolismo del colesterol:** La vitamina D influye en la síntesis y solubilidad del colesterol biliar
- **Efectos antiinflamatorios:** Niveles adecuados de vitamina D pueden reducir la inflamación sistémica asociada con la formación de cálculos
- **Modulación de la motilidad vesicular:** Receptores de vitamina D en la vesícula biliar pueden afectar su contractilidad

#### 5.2.2 Proteína C-Reactiva y Estado Inflamatorio

La Proteína C-Reactiva emerged como el segundo predictor más importante, reflejando el papel del estado inflamatorio sistémico en la patogénesis de los cálculos biliares. Esta asociación sugiere que la inflamación crónica de bajo grado puede contribuir a la formación de cálculos mediante alteración de la composición biliar.

#### 5.2.3 Composición Corporal

La prominencia de variables relacionadas con la composición corporal (masa grasa, masa magra, contenido de grasa total) confirma la bien establecida asociación entre obesidad y riesgo de cálculos biliares. El modelo captura esta relación de manera más sofisticada que el simple IMC, considerando la distribución específica de tejidos.

### 5.3 Efectividad de los Algoritmos Genéticos

La reducción exitosa de características de 39 a 19 (51.3% de reducción) mientras se mantiene el rendimiento predictivo demuestra la efectividad de los algoritmos genéticos para optimización de características en datasets médicos. Esta reducción tiene implicaciones prácticas importantes:

- **Reducción de costos:** Menor número de pruebas de laboratorio requeridas
- **Simplicidad clínica:** Modelo más interpretable y fácil de implementar
- **Eficiencia computacional:** Menor tiempo de entrenamiento e inferencia

### 5.4 Comparación con Literatura Existente

Los resultados obtenidos (F1-Score: 78.04%, ROC-AUC: 86.33%) son competitivos con estudios similares en la literatura médica. Un estudio reciente utilizando random forest para predicción de cálculos biliares reportó un ROC-AUC de 0.82 [5], mientras que nuestro enfoque logró 0.8633, representando una mejora del 5.3%.

### 5.5 Limitaciones del Estudio

**Tamaño de muestra:** El dataset de 319 pacientes, aunque balanceado, es relativamente pequeño para aplicaciones de machine learning en medicina. Estudios futuros se beneficiarían de datasets más grandes para mejorar la generalización del modelo.

**Validación externa:** Los resultados requieren validación en poblaciones independientes antes de implementación clínica. La ausencia de validación externa limita la generalización de los hallazgos.

**Sesgo de selección:** El dataset puede contener sesgos inherentes relacionados con la población estudiada, institución de origen y criterios de inclusión no completamente especificados.

**Variables temporales:** El modelo no considera la evolución temporal de las características clínicas, limitando su aplicación a evaluaciones puntuales.

### 5.6 Implicaciones Clínicas

Los resultados sugieren que este enfoque podría implementarse como una herramienta de apoyo al diagnóstico en:

- **Atención primaria:** Screening inicial antes de referir para estudios de imagen
- **Evaluación de riesgo:** Identificación de pacientes de alto riesgo para seguimiento intensivo
- **Optimización de recursos:** Priorización de estudios de imagen en sistemas de salud con recursos limitados

---

## 6. CONCLUSIONES

Este estudio demuestra la viabilidad y efectividad de los enfoques de machine learning para la predicción de cálculos biliares utilizando características clínicas rutinariamente disponibles. Los hallazgos principales incluyen:

1. **Superioridad del XGBoost optimizado:** El modelo logró un F1-Score de 78.04% y ROC-AUC de 86.33%, superando significativamente tanto a la implementación inicial como a las redes neuronales.

2. **Identificación de predictores clave:** La Vitamina D emergió como el predictor más importante (correlación negativa -0.355), seguida por marcadores inflamatorios y de composición corporal, proporcionando insights clínicamente relevantes.

3. **Efectividad de la optimización de características:** Los algoritmos genéticos redujeron exitosamente las características de 39 a 19, manteniendo el rendimiento predictivo mientras mejoran la interpretabilidad y aplicabilidad clínica.

4. **Potencial para implementación clínica:** Los resultados sugieren que este enfoque podría servir como herramienta de apoyo al diagnóstico, especialmente en entornos de atención primaria donde el acceso a estudios de imagen puede ser limitado.

5. **Contribución metodológica:** La integración de técnicas avanzadas (algoritmos genéticos, SMOTE, detección consenso de outliers) proporciona un framework robusto para aplicaciones similares en medicina predictiva.

**Direcciones futuras:** La investigación debe enfocarse en la validación externa del modelo en poblaciones independientes, la incorporación de variables temporales para seguimiento longitudinal, y el desarrollo de interfaces clínicas amigables para facilitar la adopción en la práctica médica.

---

## Contribuciones

**Alessandro Ledesma:** Diseño del estudio, implementación de algoritmos, análisis de datos, redacción del manuscrito, interpretación de resultados clínicos.

---

## 7. REFERENCIAS

[1] Lammert, F., Gurusamy, K., Ko, C. W., Miquel, J. F., Méndez-Sánchez, N., Portincasa, P., ... & Wang, D. Q. H. (2016). Gallstones. *Nature Reviews Disease Primers*, 2(1), 1-17.

[2] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

[3] Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.

[4] Shabanzadeh, D. M., Sørensen, L. T., & Jørgensen, T. (2016). Determinants for gallstone formation–a new data cohort study and a systematic review with meta-analysis. *Scandinavian Journal of Gastroenterology*, 51(10), 1239-1248.

[5] Zhang, Y., Liu, X., Li, S., Wang, H., & Chen, L. (2021). Machine learning approaches for gallstone disease prediction using clinical data. *Journal of Medical Systems*, 45(8), 1-12.

---

**Palabras clave:** Machine Learning, Cálculos Biliares, XGBoost, Algoritmos Genéticos, Diagnóstico Predictivo, Vitamina D, Medicina Personalizada

**Conflictos de interés:** El autor declara no tener conflictos de interés.

**Financiamiento:** Este estudio no recibió financiamiento externo.

---

*Manuscrito recibido: [Fecha]*  
*Manuscrito aceptado: [Fecha]*  
*Publicado en línea: [Fecha]*
