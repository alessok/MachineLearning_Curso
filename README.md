# 🧬 Proyecto de Machine Learning: Predicción de Cálculos Biliares

## 📋 Descripción del Proyecto

Este proyecto implementa técnicas avanzadas de machine learning para la predicción de cálculos biliares utilizando datos clínicos. El proyecto combina algoritmos genéticos, XGBoost y redes neuronales para crear un sistema de predicción robusto con aplicaciones médicas.

## 🎯 Objetivos

- Desarrollar un modelo predictivo para cálculos biliares usando datos clínicos
- Implementar algoritmos genéticos para selección óptima de características
- Comparar el rendimiento de XGBoost vs Redes Neuronales
- Analizar consideraciones éticas en aplicaciones médicas de ML

## 🚀 Características Destacadas

### Técnicas Implementadas:
- 🧬 **Algoritmos Genéticos** para feature selection
- ⚖️ **Técnicas de Balanceo** (SMOTE, ADASYN)
- 🕵️ **Detección Múltiple de Outliers** (4 métodos en consenso)
- 📊 **Feature Engineering** avanzado
- 🏆 **XGBoost** con optimización de hiperparámetros
- 🧠 **Redes Neuronales** (MLP) con regularización

### Resultados Alcanzados:
- **F1-Score:** 0.7804 (78.04%)
- **ROC-AUC:** 0.8506 (85.06%)
- **Reducción de características:** 60% (50 → 20 features)
- **Dataset:** 319 muestras con 39 características clínicas

## 📁 Estructura del Proyecto

```
proyecto-machine-learning/
├── 📄 Reporte_Proyecto_Gallstone_ML.md     # Reporte académico completo
├── 📊 notebooks/
│   └── proyecto_gallstone_prediction.ipynb  # Notebook principal con implementación
├── 📂 data/
│   ├── gallstone.csv                        # Dataset de cálculos biliares
│   └── Proyecto_enunciado.txt              # Especificaciones del proyecto
└── 📋 README.md                            # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Machine Learning:** scikit-learn, XGBoost
- **Algoritmos Genéticos:** DEAP
- **Balanceo de Datos:** imbalanced-learn (SMOTE)
- **Visualización:** matplotlib, seaborn
- **Análisis de Datos:** pandas, numpy
- **Desarrollo:** Jupyter Notebook

## 📊 Metodología

1. **Análisis Exploratorio de Datos (EDA)**
   - Análisis de correlaciones y distribuciones
   - Visualización de patrones demográficos
   - Identificación de variables predictoras clave

2. **Preprocesamiento Avanzado**
   - Detección múltiple de outliers (Isolation Forest, LOF, Elliptic Envelope, Z-Score)
   - Feature engineering con índices compuestos de salud
   - Normalización y escalado de características

3. **Selección de Características con Algoritmos Genéticos**
   - Población: 30 individuos, 15 generaciones
   - Fitness basado en F1-Score con penalización por número de features
   - Reducción del 60% en características manteniendo rendimiento

4. **Modelado y Evaluación**
   - XGBoost con optimización de hiperparámetros
   - Red Neuronal (MLP) con early stopping
   - Validación cruzada estratificada

## 🏆 Resultados Principales

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **XGBoost Inicial** | **0.7812** | **0.7857** | **0.7812** | **0.7804** | **0.8506** |
| XGBoost Optimizado | 0.7656 | 0.7659 | 0.7656 | 0.7656 | 0.8564 |
| Red Neuronal | 0.6562 | 0.6619 | 0.6562 | 0.6532 | 0.7227 |

### Variables Más Importantes:
1. **Vitamina D** (correlación: -0.355)
2. **Proteína C-Reactiva** (correlación: +0.282)
3. **Masa Magra** (correlación: -0.226)
4. **Edad** y **Diabetes Mellitus**

## ⚖️ Consideraciones Éticas

El proyecto incluye un análisis exhaustivo de las implicaciones éticas del uso de ML en medicina:

- **Impacto Social Positivo:** Detección temprana, reducción de costos, democratización del diagnóstico
- **Consideraciones Críticas:** Falsos positivos/negativos, privacidad de datos médicos
- **Lineamientos Éticos:** Supervisión médica, transparencia algorítmica, equidad en salud
- **Regulación:** Cumplimiento con HIPAA, GDPR, y protocolos médicos

## 🚀 Instalación y Uso

### Prerequisitos:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn deap
```

### Ejecución:
1. Clona este repositorio
2. Abre `notebooks/proyecto_gallstone_prediction.ipynb` en Jupyter
3. Ejecuta todas las celdas secuencialmente
4. Revisa el reporte completo en `Reporte_Proyecto_Gallstone_ML.md`

## 📚 Documentación

- **[Reporte Académico Completo](Reporte_Proyecto_Gallstone_ML.md):** Análisis detallado con metodología, resultados y consideraciones éticas
- **[Notebook Principal](notebooks/proyecto_gallstone_prediction.ipynb):** Implementación completa con código comentado y visualizaciones

## 👤 Autor

**Alessandro Ledesma**  
Curso de Machine Learning Avanzado  
📧 [Email de contacto]  
🔗 [LinkedIn]

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Dataset: UCI Machine Learning Repository
- Librerías: scikit-learn, XGBoost, DEAP, imbalanced-learn
- Inspiración: Aplicaciones éticas de ML en medicina

---

## 🏷️ Tags

`machine-learning` `xgboost` `genetic-algorithms` `healthcare` `python` `data-science` `medical-prediction` `feature-selection` `ethical-ai` `smote` `neural-networks` `gallstone-prediction`
