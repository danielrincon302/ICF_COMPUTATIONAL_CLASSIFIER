"""
================================================================================
CLASIFICADOR DE DISCAPACIDAD BASADO EN LA CIF
================================================================================
Sistema de clasificación computacional para la predicción de:
1. Categoría de discapacidad (física, sensorial, intelectual, psicosocial, múltiple)
2. Nivel de apoyo requerido (mínimo, intermitente, limitado, extenso)

Basado en la Clasificación Internacional del Funcionamiento, de la Discapacidad
y de la Salud (CIF) de la OMS.

Autor: Sistema de IA
Fecha: 2024
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import warnings
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================================
# CLASE PRINCIPAL DEL CLASIFICADOR
# ============================================================================

class ClasificadorDiscapacidadCIF:
    """
    Clasificador de discapacidad basado en la CIF.

    Problemas de clasificación:
    1. Categoría de discapacidad (multiclase)
    2. Nivel de apoyo requerido (multiclase ordinal)

    Características utilizadas:
    - Funciones corporales (b1-b8)
    - Estructuras corporales (s1-s8)
    - Actividades y participación (d1-d9)
    - Factores ambientales (e1-e5)
    - Variables demográficas (edad, sexo)
    """

    def __init__(self, dataset_path=None):
        """Inicializa el clasificador."""
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y_categoria = None
        self.y_apoyo = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder_categoria = LabelEncoder()
        self.label_encoder_apoyo = LabelEncoder()
        self.modelos = {}
        self.resultados = {}

    def cargar_datos(self, path=None):
        """Carga el dataset CIF."""
        if path:
            self.dataset_path = path

        print("=" * 70)
        print("CARGANDO DATASET CIF")
        print("=" * 70)

        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset cargado: {len(self.df)} registros, {len(self.df.columns)} columnas")

        return self

    def preparar_features(self):
        """Prepara las características para el modelo."""
        print("\n" + "=" * 70)
        print("PREPARANDO CARACTERÍSTICAS (FEATURES)")
        print("=" * 70)

        # Seleccionar columnas de calificadores numéricos
        calificador_cols = [col for col in self.df.columns if 'calificador' in col]

        # Añadir edad y codificar sexo
        self.df['sexo_encoded'] = (self.df['sexo'] == 'Masculino').astype(int)

        # Features finales
        feature_cols = calificador_cols + ['edad', 'sexo_encoded']
        self.feature_names = feature_cols

        self.X = self.df[feature_cols].values

        print(f"Features seleccionadas: {len(feature_cols)}")
        print(f"  - Funciones corporales (b): 8 calificadores")
        print(f"  - Estructuras corporales (s): 8 calificadores")
        print(f"  - Actividades y participación (d): 9 calificadores")
        print(f"  - Factores ambientales (e): 5 calificadores")
        print(f"  - Variables demográficas: edad, sexo")

        return self

    def preparar_targets(self):
        """Prepara las variables objetivo."""
        print("\n" + "=" * 70)
        print("PREPARANDO VARIABLES OBJETIVO (TARGETS)")
        print("=" * 70)

        # Target 1: Categoría de discapacidad (simplificada)
        def categorizar_discapacidad(tipo):
            tipo_lower = tipo.lower()
            if 'física' in tipo_lower or 'paraplejia' in tipo_lower or 'hemiplejia' in tipo_lower or 'amputación' in tipo_lower or 'artritis' in tipo_lower or 'lesión medular' in tipo_lower:
                return 'Física'
            elif 'visual' in tipo_lower or 'ceguera' in tipo_lower:
                return 'Visual'
            elif 'auditiva' in tipo_lower or 'sordera' in tipo_lower or 'hipoacusia' in tipo_lower:
                return 'Auditiva'
            elif 'intelectual' in tipo_lower:
                return 'Intelectual'
            elif 'psicosocial' in tipo_lower or 'esquizofrenia' in tipo_lower or 'bipolar' in tipo_lower or 'depresión' in tipo_lower:
                return 'Psicosocial'
            elif 'múltiple' in tipo_lower or 'parálisis cerebral' in tipo_lower or 'sordoceguera' in tipo_lower:
                return 'Múltiple'
            elif 'autista' in tipo_lower or 'autismo' in tipo_lower:
                return 'TEA'
            elif 'neurodegenerativa' in tipo_lower or 'esclerosis' in tipo_lower or 'parkinson' in tipo_lower:
                return 'Neurodegenerativa'
            elif 'distrofia' in tipo_lower or 'rara' in tipo_lower:
                return 'Enfermedad Rara'
            else:
                return 'Otra'

        self.df['categoria_discapacidad'] = self.df['tipo_discapacidad'].apply(categorizar_discapacidad)

        # Codificar targets
        self.y_categoria = self.label_encoder_categoria.fit_transform(self.df['categoria_discapacidad'])
        self.y_apoyo = self.label_encoder_apoyo.fit_transform(self.df['nivel_apoyo_requerido'])

        print("\nTarget 1 - Categoría de Discapacidad:")
        print(self.df['categoria_discapacidad'].value_counts())

        print("\nTarget 2 - Nivel de Apoyo Requerido:")
        print(self.df['nivel_apoyo_requerido'].value_counts())

        return self

    def escalar_features(self, X_train, X_test):
        """Escala las características."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def definir_modelos(self):
        """Define los modelos a evaluar."""
        self.modelos = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE, max_depth=5
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', random_state=RANDOM_STATE, probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=RANDOM_STATE, max_depth=10
            ),
            'Neural Network (MLP)': MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE
            )
        }
        return self

    def entrenar_evaluar_modelos(self, X, y, target_name, label_encoder):
        """Entrena y evalúa todos los modelos para un target específico."""
        print(f"\n{'=' * 70}")
        print(f"ENTRENAMIENTO Y EVALUACIÓN - {target_name.upper()}")
        print("=" * 70)

        # Verificar si hay clases con pocos miembros
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        # División de datos (sin stratify si hay clases muy pequeñas)
        if min_count < 2:
            print(f"⚠️  Clase minoritaria con {min_count} muestra(s). Usando división sin estratificación.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )

        # Escalar
        X_train_scaled, X_test_scaled = self.escalar_features(X_train, X_test)

        resultados = {}
        mejor_modelo = None
        mejor_f1 = 0

        print(f"\nConjunto de entrenamiento: {len(X_train)} muestras")
        print(f"Conjunto de prueba: {len(X_test)} muestras")
        print(f"Clases: {list(label_encoder.classes_)}")
        print("\n" + "-" * 70)

        for nombre, modelo in self.modelos.items():
            print(f"\n>> Entrenando: {nombre}")

            # Entrenar
            modelo.fit(X_train_scaled, y_train)

            # Predecir
            y_pred = modelo.predict(X_test_scaled)

            # Validación cruzada (ajustar folds si hay pocas muestras por clase)
            n_splits = min(CV_FOLDS, min(np.bincount(y_train)))
            n_splits = max(2, n_splits)  # Mínimo 2 folds
            cv_scores = cross_val_score(
                modelo, X_train_scaled, y_train, cv=n_splits, scoring='f1_weighted'
            )

            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            resultados[nombre] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'modelo': modelo
            }

            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Precisión: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = nombre

        print("\n" + "=" * 70)
        print(f"🏆 MEJOR MODELO: {mejor_modelo} (F1: {mejor_f1:.4f})")
        print("=" * 70)

        # Reporte detallado del mejor modelo
        print(f"\nReporte de clasificación - {mejor_modelo}:")
        # Obtener clases presentes en test
        clases_presentes = np.unique(np.concatenate([
            resultados[mejor_modelo]['y_test'],
            resultados[mejor_modelo]['y_pred']
        ]))
        nombres_clases = [label_encoder.classes_[i] for i in clases_presentes]
        print(classification_report(
            resultados[mejor_modelo]['y_test'],
            resultados[mejor_modelo]['y_pred'],
            labels=clases_presentes,
            target_names=nombres_clases,
            zero_division=0
        ))

        return resultados, mejor_modelo, (X_train_scaled, X_test_scaled, y_train, y_test)

    def obtener_importancia_features(self, modelo, nombre_modelo):
        """Obtiene la importancia de las características."""
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
        elif hasattr(modelo, 'coef_'):
            importancias = np.abs(modelo.coef_).mean(axis=0)
        else:
            return None

        df_importancia = pd.DataFrame({
            'feature': self.feature_names,
            'importancia': importancias
        }).sort_values('importancia', ascending=False)

        return df_importancia

    def visualizar_resultados(self, resultados_categoria, resultados_apoyo,
                              mejor_cat, mejor_apoyo, datos_cat, datos_apoyo):
        """Genera visualizaciones de los resultados."""
        print("\n" + "=" * 70)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 70)

        fig = plt.figure(figsize=(18, 14))

        # 1. Comparación de modelos - Categoría
        ax1 = fig.add_subplot(2, 3, 1)
        modelos_nombres = list(resultados_categoria.keys())
        f1_scores = [resultados_categoria[m]['f1_score'] for m in modelos_nombres]
        colors = ['#2ecc71' if m == mejor_cat else '#3498db' for m in modelos_nombres]
        bars = ax1.barh(range(len(modelos_nombres)), f1_scores, color=colors)
        ax1.set_yticks(range(len(modelos_nombres)))
        ax1.set_yticklabels(modelos_nombres, fontsize=9)
        ax1.set_xlabel('F1-Score')
        ax1.set_title('Comparación de Modelos\n(Categoría de Discapacidad)', fontweight='bold')
        ax1.set_xlim(0, 1)
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax1.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=8)

        # 2. Comparación de modelos - Apoyo
        ax2 = fig.add_subplot(2, 3, 2)
        f1_scores_apoyo = [resultados_apoyo[m]['f1_score'] for m in modelos_nombres]
        colors_apoyo = ['#2ecc71' if m == mejor_apoyo else '#e74c3c' for m in modelos_nombres]
        bars2 = ax2.barh(range(len(modelos_nombres)), f1_scores_apoyo, color=colors_apoyo)
        ax2.set_yticks(range(len(modelos_nombres)))
        ax2.set_yticklabels(modelos_nombres, fontsize=9)
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Comparación de Modelos\n(Nivel de Apoyo)', fontweight='bold')
        ax2.set_xlim(0, 1)
        for i, (bar, score) in enumerate(zip(bars2, f1_scores_apoyo)):
            ax2.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=8)

        # 3. Matriz de confusión - Categoría
        ax3 = fig.add_subplot(2, 3, 3)
        cm_cat = confusion_matrix(
            resultados_categoria[mejor_cat]['y_test'],
            resultados_categoria[mejor_cat]['y_pred']
        )
        im3 = ax3.imshow(cm_cat, cmap='Blues')
        clases_cat = self.label_encoder_categoria.classes_
        ax3.set_xticks(range(len(clases_cat)))
        ax3.set_yticks(range(len(clases_cat)))
        ax3.set_xticklabels(clases_cat, rotation=45, ha='right', fontsize=7)
        ax3.set_yticklabels(clases_cat, fontsize=7)
        for i in range(len(clases_cat)):
            for j in range(len(clases_cat)):
                ax3.text(j, i, cm_cat[i, j], ha='center', va='center',
                        color='white' if cm_cat[i, j] > cm_cat.max()/2 else 'black', fontsize=8)
        ax3.set_title(f'Matriz de Confusión\n{mejor_cat} (Categoría)', fontweight='bold')
        ax3.set_xlabel('Predicción')
        ax3.set_ylabel('Real')

        # 4. Importancia de features - Categoría
        ax4 = fig.add_subplot(2, 3, 4)
        modelo_cat = resultados_categoria[mejor_cat]['modelo']
        df_imp = self.obtener_importancia_features(modelo_cat, mejor_cat)
        if df_imp is not None:
            top_features = df_imp.head(15)
            colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            ax4.barh(range(len(top_features)), top_features['importancia'].values, color=colors_imp)
            ax4.set_yticks(range(len(top_features)))
            labels = [f.replace('_calificador', '').replace('_', ' ')[:20] for f in top_features['feature']]
            ax4.set_yticklabels(labels, fontsize=7)
            ax4.set_xlabel('Importancia')
            ax4.set_title(f'Top 15 Features Importantes\n{mejor_cat}', fontweight='bold')
            ax4.invert_yaxis()

        # 5. Validación cruzada
        ax5 = fig.add_subplot(2, 3, 5)
        cv_means = [resultados_categoria[m]['cv_mean'] for m in modelos_nombres]
        cv_stds = [resultados_categoria[m]['cv_std'] for m in modelos_nombres]
        x_pos = np.arange(len(modelos_nombres))
        bars5 = ax5.bar(x_pos, cv_means, yerr=cv_stds, capsize=3,
                       color=plt.cm.Spectral(np.linspace(0.2, 0.8, len(modelos_nombres))), alpha=0.8)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([m[:12] for m in modelos_nombres], rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('F1-Score (CV)')
        ax5.set_title('Validación Cruzada (5-fold)\nCategoría de Discapacidad', fontweight='bold')
        ax5.set_ylim(0, 1)

        # 6. Métricas del mejor modelo
        ax6 = fig.add_subplot(2, 3, 6)
        metricas = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        valores_cat = [
            resultados_categoria[mejor_cat]['accuracy'],
            resultados_categoria[mejor_cat]['f1_score'],
            resultados_categoria[mejor_cat]['precision'],
            resultados_categoria[mejor_cat]['recall']
        ]
        valores_apoyo = [
            resultados_apoyo[mejor_apoyo]['accuracy'],
            resultados_apoyo[mejor_apoyo]['f1_score'],
            resultados_apoyo[mejor_apoyo]['precision'],
            resultados_apoyo[mejor_apoyo]['recall']
        ]
        x = np.arange(len(metricas))
        width = 0.35
        bars_cat = ax6.bar(x - width/2, valores_cat, width, label=f'Categoría ({mejor_cat})', color='#3498db')
        bars_apoyo = ax6.bar(x + width/2, valores_apoyo, width, label=f'Apoyo ({mejor_apoyo})', color='#e74c3c')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metricas)
        ax6.set_ylabel('Score')
        ax6.set_title('Métricas de los Mejores Modelos', fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.set_ylim(0, 1)
        for bar, val in zip(bars_cat, valores_cat):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
        for bar, val in zip(bars_apoyo, valores_apoyo):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('resultados_clasificador_cif.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Visualización guardada: resultados_clasificador_cif.png")

        return fig

    def guardar_modelo(self, modelo, nombre, target):
        """Guarda el modelo entrenado."""
        filename = f'modelo_{target}_{nombre.lower().replace(" ", "_")}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump({
                'modelo': modelo,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'label_encoder': self.label_encoder_categoria if target == 'categoria' else self.label_encoder_apoyo
            }, f)
        print(f"Modelo guardado: {filename}")
        return filename

    def generar_reporte(self, resultados_categoria, resultados_apoyo, mejor_cat, mejor_apoyo):
        """Genera un reporte JSON con los resultados."""
        reporte = {
            'fecha_generacion': datetime.now().isoformat(),
            'dataset': {
                'total_muestras': len(self.df),
                'total_features': len(self.feature_names)
            },
            'problema_1_categoria': {
                'clases': list(self.label_encoder_categoria.classes_),
                'mejor_modelo': mejor_cat,
                'metricas_mejor': {
                    'accuracy': float(resultados_categoria[mejor_cat]['accuracy']),
                    'f1_score': float(resultados_categoria[mejor_cat]['f1_score']),
                    'precision': float(resultados_categoria[mejor_cat]['precision']),
                    'recall': float(resultados_categoria[mejor_cat]['recall']),
                    'cv_f1_mean': float(resultados_categoria[mejor_cat]['cv_mean']),
                    'cv_f1_std': float(resultados_categoria[mejor_cat]['cv_std'])
                },
                'todos_los_modelos': {
                    nombre: {
                        'f1_score': float(res['f1_score']),
                        'accuracy': float(res['accuracy'])
                    } for nombre, res in resultados_categoria.items()
                }
            },
            'problema_2_apoyo': {
                'clases': list(self.label_encoder_apoyo.classes_),
                'mejor_modelo': mejor_apoyo,
                'metricas_mejor': {
                    'accuracy': float(resultados_apoyo[mejor_apoyo]['accuracy']),
                    'f1_score': float(resultados_apoyo[mejor_apoyo]['f1_score']),
                    'precision': float(resultados_apoyo[mejor_apoyo]['precision']),
                    'recall': float(resultados_apoyo[mejor_apoyo]['recall']),
                    'cv_f1_mean': float(resultados_apoyo[mejor_apoyo]['cv_mean']),
                    'cv_f1_std': float(resultados_apoyo[mejor_apoyo]['cv_std'])
                }
            }
        }

        with open('reporte_clasificador_cif.json', 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)

        print("Reporte guardado: reporte_clasificador_cif.json")
        return reporte

    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de clasificación."""
        print("\n" + "=" * 70)
        print("   CLASIFICADOR DE DISCAPACIDAD CIF - PIPELINE COMPLETO")
        print("=" * 70)

        # 1. Cargar y preparar datos
        self.cargar_datos()
        self.preparar_features()
        self.preparar_targets()
        self.definir_modelos()

        # 2. Entrenar modelos para categoría de discapacidad
        resultados_categoria, mejor_cat, datos_cat = self.entrenar_evaluar_modelos(
            self.X, self.y_categoria,
            "Categoría de Discapacidad",
            self.label_encoder_categoria
        )

        # 3. Entrenar modelos para nivel de apoyo
        resultados_apoyo, mejor_apoyo, datos_apoyo = self.entrenar_evaluar_modelos(
            self.X, self.y_apoyo,
            "Nivel de Apoyo Requerido",
            self.label_encoder_apoyo
        )

        # 4. Visualizar resultados
        self.visualizar_resultados(
            resultados_categoria, resultados_apoyo,
            mejor_cat, mejor_apoyo,
            datos_cat, datos_apoyo
        )

        # 5. Guardar mejores modelos
        self.guardar_modelo(resultados_categoria[mejor_cat]['modelo'], mejor_cat, 'categoria')
        self.guardar_modelo(resultados_apoyo[mejor_apoyo]['modelo'], mejor_apoyo, 'apoyo')

        # 6. Generar reporte
        reporte = self.generar_reporte(resultados_categoria, resultados_apoyo, mejor_cat, mejor_apoyo)

        print("\n" + "=" * 70)
        print("   PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 70)

        return {
            'resultados_categoria': resultados_categoria,
            'resultados_apoyo': resultados_apoyo,
            'mejor_modelo_categoria': mejor_cat,
            'mejor_modelo_apoyo': mejor_apoyo,
            'reporte': reporte
        }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Crear instancia del clasificador
    clasificador = ClasificadorDiscapacidadCIF(
        dataset_path='dataset_cif_discapacidad_100.csv'
    )

    # Ejecutar pipeline completo
    resultados = clasificador.ejecutar_pipeline_completo()

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\n📊 Mejor modelo para Categoría: {resultados['mejor_modelo_categoria']}")
    print(f"   F1-Score: {resultados['reporte']['problema_1_categoria']['metricas_mejor']['f1_score']:.4f}")
    print(f"\n📊 Mejor modelo para Nivel de Apoyo: {resultados['mejor_modelo_apoyo']}")
    print(f"   F1-Score: {resultados['reporte']['problema_2_apoyo']['metricas_mejor']['f1_score']:.4f}")
    print("\nArchivos generados:")
    print("  - resultados_clasificador_cif.png")
    print("  - reporte_clasificador_cif.json")
    print("  - modelo_categoria_*.pkl")
    print("  - modelo_apoyo_*.pkl")
