"""
Generador de Dataset Simulado CIF (Clasificación Internacional del Funcionamiento)
================================================================================
Este script genera un dataset de 100 personas con discapacidad utilizando
la estructura de la CIF para la evaluación de la salud y la discapacidad.

Calificadores CIF:
- 0: Sin problema (0-4%)
- 1: Problema leve (5-24%)
- 2: Problema moderado (25-49%)
- 3: Problema grave (50-95%)
- 4: Problema completo (96-100%)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# ============================================================================
# DEFINICIÓN DE CATEGORÍAS CIF
# ============================================================================

# Funciones Corporales (b)
funciones_corporales = {
    'b1_funciones_mentales': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b2_funciones_sensoriales_dolor': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b3_funciones_voz_habla': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b4_funciones_cardiovascular_respiratorio': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b5_funciones_digestivo_metabolico_endocrino': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b6_funciones_genitourinarias_reproductivas': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b7_funciones_neuromusculoesqueleticas_movimiento': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    'b8_funciones_piel': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ]
}

# Estructuras Corporales (s)
estructuras_corporales = {
    's1_estructura_sistema_nervioso': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's2_ojo_oido_estructuras_relacionadas': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's3_estructuras_voz_habla': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's4_estructura_cardiovascular_respiratorio': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's5_estructuras_digestivo_metabolico_endocrino': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's6_estructuras_genitourinario_reproductivo': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's7_estructuras_movimiento': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ],
    's8_piel_estructuras_relacionadas': [
        'Sin deficiencia', 'Deficiencia leve', 'Deficiencia moderada',
        'Deficiencia grave', 'Deficiencia completa'
    ]
}

# Actividades y Participación (d)
actividades_participacion = {
    'd1_aprendizaje_aplicacion_conocimiento': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd2_tareas_demandas_generales': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd3_comunicacion': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd4_movilidad': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd5_autocuidado': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd6_vida_domestica': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd7_interacciones_relaciones_interpersonales': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd8_areas_principales_vida': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ],
    'd9_vida_comunitaria_social_civica': [
        'Sin limitación', 'Limitación leve', 'Limitación moderada',
        'Limitación grave', 'Limitación completa'
    ]
}

# Factores Ambientales (e)
factores_ambientales = {
    'e1_productos_tecnologia': [
        'Barrera completa', 'Barrera grave', 'Barrera moderada',
        'Barrera leve', 'Sin barrera', 'Facilitador leve',
        'Facilitador moderado', 'Facilitador sustancial', 'Facilitador completo'
    ],
    'e2_entorno_natural_cambios_humanos': [
        'Barrera completa', 'Barrera grave', 'Barrera moderada',
        'Barrera leve', 'Sin barrera', 'Facilitador leve',
        'Facilitador moderado', 'Facilitador sustancial', 'Facilitador completo'
    ],
    'e3_apoyo_relaciones': [
        'Barrera completa', 'Barrera grave', 'Barrera moderada',
        'Barrera leve', 'Sin barrera', 'Facilitador leve',
        'Facilitador moderado', 'Facilitador sustancial', 'Facilitador completo'
    ],
    'e4_actitudes': [
        'Barrera completa', 'Barrera grave', 'Barrera moderada',
        'Barrera leve', 'Sin barrera', 'Facilitador leve',
        'Facilitador moderado', 'Facilitador sustancial', 'Facilitador completo'
    ],
    'e5_servicios_sistemas_politicas': [
        'Barrera completa', 'Barrera grave', 'Barrera moderada',
        'Barrera leve', 'Sin barrera', 'Facilitador leve',
        'Facilitador moderado', 'Facilitador sustancial', 'Facilitador completo'
    ]
}

# Tipos de discapacidad principales
tipos_discapacidad = [
    'Discapacidad física - Paraplejia',
    'Discapacidad física - Hemiplejia',
    'Discapacidad física - Amputación miembro inferior',
    'Discapacidad física - Amputación miembro superior',
    'Discapacidad física - Artritis reumatoide',
    'Discapacidad visual - Ceguera total',
    'Discapacidad visual - Baja visión',
    'Discapacidad auditiva - Sordera profunda',
    'Discapacidad auditiva - Hipoacusia',
    'Discapacidad intelectual - Leve',
    'Discapacidad intelectual - Moderada',
    'Discapacidad intelectual - Grave',
    'Discapacidad psicosocial - Esquizofrenia',
    'Discapacidad psicosocial - Trastorno bipolar',
    'Discapacidad psicosocial - Depresión mayor',
    'Discapacidad múltiple - Parálisis cerebral',
    'Discapacidad múltiple - Sordoceguera',
    'Trastorno del espectro autista',
    'Enfermedad neurodegenerativa - Esclerosis múltiple',
    'Enfermedad neurodegenerativa - Parkinson',
    'Lesión medular - Tetraplejia',
    'Enfermedad rara - Distrofia muscular'
]

# Etiología/Origen de la discapacidad
etiologias = [
    'Congénita', 'Adquirida - Accidente', 'Adquirida - Enfermedad',
    'Adquirida - Trauma', 'Perinatal', 'Genética', 'Desconocida'
]

# Nombres ficticios
nombres_masculinos = [
    'Juan', 'Carlos', 'Miguel', 'José', 'Luis', 'Pedro', 'Antonio', 'Francisco',
    'Manuel', 'Diego', 'Andrés', 'Fernando', 'Ricardo', 'Eduardo', 'Gabriel',
    'Sergio', 'Pablo', 'Javier', 'Alberto', 'Roberto', 'Alejandro', 'Daniel',
    'Martín', 'Raúl', 'Héctor', 'Óscar', 'Enrique', 'Víctor', 'Hugo', 'Iván'
]

nombres_femeninos = [
    'María', 'Ana', 'Carmen', 'Laura', 'Patricia', 'Sofía', 'Isabel', 'Lucía',
    'Elena', 'Rosa', 'Claudia', 'Marta', 'Andrea', 'Paula', 'Sara', 'Diana',
    'Gabriela', 'Valentina', 'Camila', 'Daniela', 'Natalia', 'Julia', 'Mónica',
    'Adriana', 'Beatriz', 'Carolina', 'Silvia', 'Teresa', 'Verónica', 'Lorena'
]

apellidos = [
    'García', 'Rodríguez', 'Martínez', 'López', 'González', 'Hernández', 'Pérez',
    'Sánchez', 'Ramírez', 'Torres', 'Flores', 'Rivera', 'Gómez', 'Díaz', 'Cruz',
    'Morales', 'Reyes', 'Ortiz', 'Gutiérrez', 'Mendoza', 'Ruiz', 'Jiménez',
    'Vargas', 'Rojas', 'Castillo', 'Medina', 'Castro', 'Herrera', 'Vega', 'Ramos'
]

# ============================================================================
# FUNCIONES DE GENERACIÓN
# ============================================================================

def generar_calificador_ponderado(pesos):
    """Genera un calificador basado en pesos específicos"""
    opciones = list(range(len(pesos)))
    return np.random.choice(opciones, p=pesos)

def generar_perfil_discapacidad(tipo_discapacidad):
    """
    Genera un perfil coherente de calificadores CIF basado en el tipo de discapacidad.
    Esto asegura que los datos sean realistas y coherentes.
    """
    perfil = {}

    # Definir pesos base según tipo de discapacidad
    if 'física' in tipo_discapacidad.lower() or 'paraplejia' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b7_funciones_neuromusculoesqueleticas_movimiento']
        perfil['estructuras_principales'] = ['s7_estructuras_movimiento', 's1_estructura_sistema_nervioso']
        perfil['actividades_principales'] = ['d4_movilidad', 'd5_autocuidado', 'd6_vida_domestica']
        perfil['peso_deficiencia'] = [0.05, 0.15, 0.30, 0.35, 0.15]  # Más hacia grave

    elif 'visual' in tipo_discapacidad.lower() or 'ceguera' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b2_funciones_sensoriales_dolor']
        perfil['estructuras_principales'] = ['s2_ojo_oido_estructuras_relacionadas']
        perfil['actividades_principales'] = ['d1_aprendizaje_aplicacion_conocimiento', 'd4_movilidad', 'd3_comunicacion']
        perfil['peso_deficiencia'] = [0.05, 0.10, 0.25, 0.35, 0.25]

    elif 'auditiva' in tipo_discapacidad.lower() or 'sordera' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b2_funciones_sensoriales_dolor', 'b3_funciones_voz_habla']
        perfil['estructuras_principales'] = ['s2_ojo_oido_estructuras_relacionadas', 's3_estructuras_voz_habla']
        perfil['actividades_principales'] = ['d3_comunicacion', 'd7_interacciones_relaciones_interpersonales']
        perfil['peso_deficiencia'] = [0.05, 0.15, 0.30, 0.30, 0.20]

    elif 'intelectual' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b1_funciones_mentales']
        perfil['estructuras_principales'] = ['s1_estructura_sistema_nervioso']
        perfil['actividades_principales'] = ['d1_aprendizaje_aplicacion_conocimiento', 'd2_tareas_demandas_generales',
                                             'd7_interacciones_relaciones_interpersonales', 'd8_areas_principales_vida']
        if 'leve' in tipo_discapacidad.lower():
            perfil['peso_deficiencia'] = [0.10, 0.40, 0.35, 0.10, 0.05]
        elif 'moderada' in tipo_discapacidad.lower():
            perfil['peso_deficiencia'] = [0.05, 0.20, 0.45, 0.25, 0.05]
        else:
            perfil['peso_deficiencia'] = [0.05, 0.10, 0.25, 0.40, 0.20]

    elif 'psicosocial' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b1_funciones_mentales']
        perfil['estructuras_principales'] = ['s1_estructura_sistema_nervioso']
        perfil['actividades_principales'] = ['d7_interacciones_relaciones_interpersonales', 'd8_areas_principales_vida',
                                             'd9_vida_comunitaria_social_civica', 'd2_tareas_demandas_generales']
        perfil['peso_deficiencia'] = [0.10, 0.25, 0.35, 0.25, 0.05]

    elif 'autista' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b1_funciones_mentales']
        perfil['estructuras_principales'] = ['s1_estructura_sistema_nervioso']
        perfil['actividades_principales'] = ['d3_comunicacion', 'd7_interacciones_relaciones_interpersonales',
                                             'd9_vida_comunitaria_social_civica']
        perfil['peso_deficiencia'] = [0.10, 0.30, 0.35, 0.20, 0.05]

    elif 'múltiple' in tipo_discapacidad.lower() or 'parálisis cerebral' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b1_funciones_mentales', 'b7_funciones_neuromusculoesqueleticas_movimiento',
                                           'b3_funciones_voz_habla']
        perfil['estructuras_principales'] = ['s1_estructura_sistema_nervioso', 's7_estructuras_movimiento']
        perfil['actividades_principales'] = ['d4_movilidad', 'd3_comunicacion', 'd5_autocuidado',
                                             'd1_aprendizaje_aplicacion_conocimiento']
        perfil['peso_deficiencia'] = [0.05, 0.15, 0.25, 0.35, 0.20]

    elif 'neurodegenerativa' in tipo_discapacidad.lower():
        perfil['funciones_principales'] = ['b1_funciones_mentales', 'b7_funciones_neuromusculoesqueleticas_movimiento']
        perfil['estructuras_principales'] = ['s1_estructura_sistema_nervioso', 's7_estructuras_movimiento']
        perfil['actividades_principales'] = ['d4_movilidad', 'd5_autocuidado', 'd2_tareas_demandas_generales']
        perfil['peso_deficiencia'] = [0.10, 0.20, 0.35, 0.25, 0.10]

    else:  # Perfil genérico
        perfil['funciones_principales'] = ['b7_funciones_neuromusculoesqueleticas_movimiento']
        perfil['estructuras_principales'] = ['s7_estructuras_movimiento']
        perfil['actividades_principales'] = ['d4_movilidad', 'd5_autocuidado']
        perfil['peso_deficiencia'] = [0.15, 0.25, 0.30, 0.20, 0.10]

    return perfil

def generar_persona(id_persona):
    """Genera los datos de una persona con discapacidad"""

    # Datos demográficos
    sexo = random.choice(['Masculino', 'Femenino'])
    if sexo == 'Masculino':
        nombre = random.choice(nombres_masculinos)
    else:
        nombre = random.choice(nombres_femeninos)

    apellido1 = random.choice(apellidos)
    apellido2 = random.choice(apellidos)
    nombre_completo = f"{nombre} {apellido1} {apellido2}"

    # Distribución de edades: 5-85 años (81 valores)
    pesos_edad = [0.01]*10 + [0.015]*10 + [0.02]*20 + [0.025]*20 + [0.015]*15 + [0.01]*6  # 81 valores
    pesos_edad = np.array(pesos_edad) / sum(pesos_edad)
    edad = np.random.choice(range(5, 86), p=pesos_edad)

    # Tipo de discapacidad y etiología
    tipo_disc = random.choice(tipos_discapacidad)
    etiologia = random.choice(etiologias)

    # Ajustar etiología según tipo
    if 'congénita' in tipo_disc.lower() or 'genética' in tipo_disc.lower():
        etiologia = random.choice(['Congénita', 'Genética', 'Perinatal'])

    # Obtener perfil de discapacidad
    perfil = generar_perfil_discapacidad(tipo_disc)
    peso_principal = perfil['peso_deficiencia']
    peso_secundario = [0.40, 0.30, 0.20, 0.08, 0.02]  # Peso para funciones no principales

    # Generar calificadores numéricos (0-4)
    datos = {
        'id': f'PCD-{id_persona:04d}',
        'nombre_completo': nombre_completo,
        'edad': edad,
        'sexo': sexo,
        'tipo_discapacidad': tipo_disc,
        'etiologia': etiologia,
        'fecha_evaluacion': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
    }

    # Funciones Corporales (calificadores numéricos 0-4)
    for func, valores in funciones_corporales.items():
        if func in perfil.get('funciones_principales', []):
            calif = generar_calificador_ponderado(peso_principal)
        else:
            calif = generar_calificador_ponderado(peso_secundario)
        datos[f'{func}_calificador'] = calif
        datos[f'{func}_descripcion'] = valores[calif]

    # Estructuras Corporales
    for estr, valores in estructuras_corporales.items():
        if estr in perfil.get('estructuras_principales', []):
            calif = generar_calificador_ponderado(peso_principal)
        else:
            calif = generar_calificador_ponderado(peso_secundario)
        datos[f'{estr}_calificador'] = calif
        datos[f'{estr}_descripcion'] = valores[calif]

    # Actividades y Participación
    for act, valores in actividades_participacion.items():
        if act in perfil.get('actividades_principales', []):
            calif = generar_calificador_ponderado(peso_principal)
        else:
            calif = generar_calificador_ponderado(peso_secundario)
        datos[f'{act}_calificador'] = calif
        datos[f'{act}_descripcion'] = valores[calif]

    # Factores Ambientales (escala -4 a +4)
    peso_ambiental = [0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.12, 0.08, 0.05]
    for amb, valores in factores_ambientales.items():
        calif = generar_calificador_ponderado(peso_ambiental)
        datos[f'{amb}_calificador'] = calif - 4  # Convertir a escala -4 a +4
        datos[f'{amb}_descripcion'] = valores[calif]

    # Calcular índice de severidad global (promedio de calificadores principales)
    calificadores_func = [datos[f'{k}_calificador'] for k in funciones_corporales.keys()]
    calificadores_act = [datos[f'{k}_calificador'] for k in actividades_participacion.keys()]

    datos['indice_severidad_funcional'] = round(np.mean(calificadores_func), 2)
    datos['indice_limitacion_actividades'] = round(np.mean(calificadores_act), 2)
    datos['indice_global_discapacidad'] = round(
        (np.mean(calificadores_func) + np.mean(calificadores_act)) / 2, 2
    )

    # Necesidad de apoyo
    if datos['indice_global_discapacidad'] >= 3:
        datos['nivel_apoyo_requerido'] = 'Extenso'
    elif datos['indice_global_discapacidad'] >= 2:
        datos['nivel_apoyo_requerido'] = 'Limitado'
    elif datos['indice_global_discapacidad'] >= 1:
        datos['nivel_apoyo_requerido'] = 'Intermitente'
    else:
        datos['nivel_apoyo_requerido'] = 'Mínimo'

    return datos

# ============================================================================
# GENERACIÓN DEL DATASET
# ============================================================================

def generar_dataset(n_personas=100):
    """Genera el dataset completo de personas con discapacidad"""
    print(f"Generando dataset de {n_personas} personas con discapacidad...")

    datos_completos = []
    for i in range(1, n_personas + 1):
        persona = generar_persona(i)
        datos_completos.append(persona)
        if i % 20 == 0:
            print(f"  Generadas {i} personas...")

    df = pd.DataFrame(datos_completos)
    print(f"Dataset generado exitosamente con {len(df)} registros.")
    return df

if __name__ == "__main__":
    # Generar dataset
    df = generar_dataset(100)

    # Guardar en CSV
    csv_path = '/home/cesar.rincon@siesa.com/GitHubPersonal/ICF/ICF_COMPUTATIONAL_CLASSIFIER/dataset_cif_discapacidad_100.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nDataset guardado en: {csv_path}")

    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DEL DATASET GENERADO")
    print("="*70)
    print(f"Total de registros: {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"\nDistribución por sexo:")
    print(df['sexo'].value_counts())
    print(f"\nDistribución por tipo de discapacidad:")
    print(df['tipo_discapacidad'].value_counts())
    print(f"\nEstadísticas de edad:")
    print(df['edad'].describe())
    print(f"\nDistribución por nivel de apoyo requerido:")
    print(df['nivel_apoyo_requerido'].value_counts())
    print(f"\nÍndice global de discapacidad (estadísticas):")
    print(df['indice_global_discapacidad'].describe())
