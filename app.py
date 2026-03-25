#!/usr/bin/env python3
"""
AgroEco Lab - Plataforma de Análisis Agroecológico
Sistema de visualización, filtrado e IA para datos de suelo
"""

import os
import json
import math
import requests
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Cargar y limpiar dataset ────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "base_datos_agroecologica_v2.csv")
def load_data():
    df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8-sig', low_memory=False)
    # Drop junk columns
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed') or c == '859' or c == '23/11/2022'], errors='ignore')
    # Numeric columns
    num_cols = ['acidez_interc_meq','Al_meq_100g','S_ppm','B_ppm','Ca_meq_100g',
                'C_organico_pct','Cu_ppm','CE_dS_m','densidad_aparente','P_disponible_ppm',
                'Fe_ppm','humedad_higrosc_pct','Mg_meq_100g','Mn_ppm','MO_pct',
                'N_amoniacal','N_total_pct','K_meq_100g','Na_meq_100g','Zn_ppm','pH',
                'arena_pct','arcilla_pct','limo_pct','CICA_meq_100g','CICE_meq_100g']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Clean text
    for c in ['cultivo','municipio','departamento','tipo_suelo','TIPO_MUESTRA']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()
            df[c] = df[c].replace({'Nan': None, 'None': None, '': None})
    df['fecha_muestra'] = pd.to_datetime(df['fecha_muestra'], errors='coerce', dayfirst=True)
    df['año'] = df['fecha_muestra'].dt.year
    df['mes'] = df['fecha_muestra'].dt.month
    # Drop rows with no key data
    df = df.dropna(subset=['pH'])
    df = df.reset_index(drop=True)
    mask = df['tipo_suelo'].isna()
    df.loc[mask, 'tipo_suelo'] = df[mask].apply(
    lambda r: clasificar_textura(r.get('arena_pct'), r.get('limo_pct'), r.get('arcilla_pct')), axis=1
    )
    return df

df_global = load_data()

# ── Clasificación de estado ─────────────────────────────────────────────────
def classify_estado(row):
    """Clasifica cada muestra en Óptimo, Medio o Crítico basado en pH, MO y nutrientes clave."""
    score = 0
    max_score = 0

    # pH (ideal 5.5–7.0 para la mayoría de cultivos)
    if pd.notna(row.get('pH')):
        max_score += 3
        ph = row['pH']
        if 5.5 <= ph <= 7.0:
            score += 3
        elif 5.0 <= ph < 5.5 or 7.0 < ph <= 7.5:
            score += 2
        elif 4.5 <= ph < 5.0 or 7.5 < ph <= 8.0:
            score += 1

    # Materia Orgánica
    if pd.notna(row.get('MO_pct')):
        max_score += 3
        mo = row['MO_pct']
        if mo >= 5:
            score += 3
        elif mo >= 3:
            score += 2
        elif mo >= 1.5:
            score += 1

    # Fósforo disponible
    if pd.notna(row.get('P_disponible_ppm')):
        max_score += 2
        p = row['P_disponible_ppm']
        if 20 <= p <= 60:
            score += 2
        elif p > 10:
            score += 1

    # Aluminio (toxicidad)
    if pd.notna(row.get('Al_meq_100g')):
        max_score += 2
        al = row['Al_meq_100g']
        if al < 0.5:
            score += 2
        elif al < 1.5:
            score += 1

    # Conductividad eléctrica
    if pd.notna(row.get('CE_dS_m')):
        max_score += 2
        ce = row['CE_dS_m']
        if ce <= 2.0:
            score += 2
        elif ce <= 4.0:
            score += 1

    if max_score == 0:
        return 'Sin Datos'
    pct = score / max_score
    if pct >= 0.70:
        return 'Óptimo'
    elif pct >= 0.40:
        return 'Medio'
    else:
        return 'Crítico'

df_global['estado'] = df_global.apply(classify_estado, axis=1)

def clasificar_textura(arena, limo, arcilla):
    """Clasifica el tipo de suelo según el triángulo textural USDA."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [arena, limo, arcilla]):
        return None
    # Normalizar si no suman exactamente 100
    total = arena + limo + arcilla
    if total == 0:
        return None
    arena  = (arena  / total) * 100
    limo   = (limo   / total) * 100
    arcilla= (arcilla/ total) * 100

    if arcilla >= 40 and limo < 40 and arena < 45:
        return 'Arcilloso'
    elif arcilla >= 40 and limo >= 40:
        return 'Arcillo Limoso'
    elif arcilla >= 35 and arena >= 45:
        return 'Arcillo Arenoso'
    elif arcilla >= 27 and arena >= 45:
        return 'Franco Arcillo Arenoso'
    elif arcilla >= 27 and limo >= 40:
        return 'Franco Arcillo Limoso'
    elif arcilla >= 27:
        return 'Franco Arcilloso'
    elif limo >= 80 and arcilla < 12:
        return 'Limoso'
    elif limo >= 50 and arcilla < 27:
        return 'Franco Limoso'
    elif arena >= 70 and arcilla < 15:
        return 'Arena Franca'
    elif arena >= 85:
        return 'Arenoso'
    elif arena >= 55 and arcilla < 20:
        return 'Franco Arenoso'
    else:
        return 'Franco'
    
# ── ML: Predicción de Estado ────────────────────────────────────────────────
ML_FEATURES = ['pH', 'MO_pct', 'P_disponible_ppm', 'Ca_meq_100g', 'Mg_meq_100g',
               'K_meq_100g', 'Al_meq_100g', 'CE_dS_m', 'N_total_pct', 'C_organico_pct']

def train_model(df):
    dft = df[ML_FEATURES + ['estado']].dropna()
    dft = dft[dft['estado'].isin(['Óptimo','Medio','Crítico'])]
    if len(dft) < 50:
        return None, None
    le = LabelEncoder()
    y = le.fit_transform(dft['estado'])
    X = dft[ML_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, le, acc

clf_model, label_enc, model_acc = None, None, 0.0
try:
    result = train_model(df_global)
    if result and len(result) == 3:
        clf_model, label_enc, model_acc = result
except Exception as e:
    print(f"ML training skipped: {e}")

# ── Recomendaciones IA (local rule-based + Claude API opcional) ─────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def generate_local_recommendations(row):
    recs = []
    warnings_list = []
    ph = row.get('pH')
    mo = row.get('MO_pct')
    p = row.get('P_disponible_ppm')
    al = row.get('Al_meq_100g')
    ce = row.get('CE_dS_m')
    ca = row.get('Ca_meq_100g')
    mg = row.get('Mg_meq_100g')
    k = row.get('K_meq_100g')
    n = row.get('N_total_pct')
    fe = row.get('Fe_ppm')
    zn = row.get('Zn_ppm')
    mn = row.get('Mn_ppm')
    b = row.get('B_ppm')
    na = row.get('Na_meq_100g')
    da = row.get('densidad_aparente')
    arena = row.get('arena_pct')
    arcilla = row.get('arcilla_pct')
    cultivo = row.get('cultivo', '')

    # 1. pH
    if pd.notna(ph):
        if ph < 4.5:
            warnings_list.append(f"pH muy ácido ({ph:.2f}): Aplicar cal agrícola o dolomita (2-4 t/ha) para elevar el pH. Riesgo alto de toxicidad por Al y Mn.")
        elif ph < 5.5:
            recs.append(f"pH ácido ({ph:.2f}): Encalado moderado con cal dolomítica para alcanzar rango óptimo 5.5–6.5. Dosis sugerida: 1-2 t/ha según saturación de Al.")
        elif ph > 8.0:
            warnings_list.append(f"pH muy alcalino ({ph:.2f}): Aplicar azufre elemental (200-400 kg/ha) o ácido húmico. Riesgo de deficiencias de Fe, Mn, Zn y B.")
        elif ph > 7.5:
            recs.append(f"pH ligeramente alcalino ({ph:.2f}): Incorporar materia orgánica ácida o azufre para reducirlo gradualmente.")
        else:
            recs.append(f"pH óptimo ({ph:.2f}): Condición ideal para la mayoría de cultivos. Monitorear cada 6 meses.")

    # 2. Materia Orgánica
    if pd.notna(mo):
        if mo < 1.5:
            warnings_list.append(f"Materia orgánica crítica ({mo:.1f}%): Incorporar compost maduro (10-15 t/ha), vermicompost o abonos verdes de forma urgente.")
        elif mo < 3.0:
            recs.append(f"Materia orgánica baja ({mo:.1f}%): Aplicar compost (5-8 t/ha/año) y establecer coberturas vegetales para incrementar MO progresivamente.")
        elif mo < 5.0:
            recs.append(f"Materia orgánica media ({mo:.1f}%): Mantener con aportes anuales de compost o residuos de cosecha. Evitar quemas.")
        else:
            recs.append(f"Excelente nivel de materia orgánica ({mo:.1f}%): Continuar prácticas actuales. Considerar biofertilizantes para potenciar actividad microbiana.")

    # 3. Fósforo
    if pd.notna(p):
        if p < 10:
            recs.append(f"Fósforo deficiente ({p:.1f} ppm): Aplicar roca fosfórica (300-500 kg/ha) o DAP. En suelos ácidos, el encalado mejora la disponibilidad.")
        elif p < 20:
            recs.append(f"Fósforo bajo ({p:.1f} ppm): Fertilización fosfórica moderada (100-200 kg/ha de superfosfato triple). Combinar con micorrizas para mejor absorción.")
        elif p > 80:
            warnings_list.append(f"Exceso de fósforo ({p:.1f} ppm): Suspender fertilización fosfórica. Puede bloquear absorción de Zn, Fe y Cu. Aplicar zinc quelado preventivamente.")
        elif p > 50:
            recs.append(f"Fósforo alto ({p:.1f} ppm): No aplicar más fósforo por esta temporada. Monitorear disponibilidad de micronutrientes.")
        else:
            recs.append(f"Fósforo en rango adecuado ({p:.1f} ppm): Mantener fertilización de mantenimiento (50-80 kg/ha/año).")

    # 4. Aluminio
    if pd.notna(al):
        if al > 2.0:
            warnings_list.append(f"Toxicidad severa por aluminio ({al:.2f} meq/100g): Encalado urgente (3-5 t/ha de cal agrícola). El Al³⁺ inhibe el crecimiento radicular gravemente.")
        elif al > 1.0:
            warnings_list.append(f"Aluminio elevado ({al:.2f} meq/100g): Aplicar cal dolomítica para precipitar el Al³⁺. Revisar saturación de Al respecto a CICE.")
        elif al > 0.5:
            recs.append(f"Aluminio moderado ({al:.2f} meq/100g): Monitorear y considerar encalado preventivo si el pH es menor a 5.5.")

    # 5. Potasio
    if pd.notna(k):
        if k < 0.2:
            warnings_list.append(f"Potasio crítico ({k:.2f} meq/100g): Aplicar KCl o K₂SO₄ (100-150 kg/ha). El K es esencial para fructificación, resistencia a sequía y enfermedades.")
        elif k < 0.4:
            recs.append(f"Potasio bajo ({k:.2f} meq/100g): Fertilización potásica moderada (60-100 kg/ha de KCl). Fraccionar en 2-3 aplicaciones.")
        elif k > 2.0:
            recs.append(f"Potasio alto ({k:.2f} meq/100g): Reducir fertilización potásica. Niveles excesivos pueden antagonizar Ca y Mg.")

    # 6. Nitrógeno
    if pd.notna(n):
        if n < 0.1:
            warnings_list.append(f"Nitrógeno total muy bajo ({n:.2f}%): Aplicar urea o sulfato de amonio fraccionado. Considerar inoculación con bacterias fijadoras (Azospirillum, Rhizobium).")
        elif n < 0.2:
            recs.append(f"Nitrógeno bajo ({n:.2f}%): Fertilización nitrogenada moderada (80-120 kg N/ha). Fraccionar para reducir pérdidas por lixiviación.")
        elif n > 0.5:
            recs.append(f"Nitrógeno alto ({n:.2f}%): Reducir aplicaciones nitrogenadas para evitar lixiviación y contaminación de fuentes hídricas.")

    # 7. Conductividad eléctrica / Salinidad
    if pd.notna(ce):
        if ce > 4.0:
            warnings_list.append(f"Salinidad alta (CE={ce:.2f} dS/m): Riego de lavado con agua de baja conductividad. Suspender fertilizantes salinos. Revisar calidad del agua de riego.")
        elif ce > 2.0:
            recs.append(f"Salinidad moderada (CE={ce:.2f} dS/m): Monitorear mensualmente, usar fertilizantes de baja salinidad (nitrato de calcio, sulfatos) y mejorar drenaje.")
        else:
            recs.append(f"Conductividad eléctrica normal (CE={ce:.2f} dS/m): Sin riesgo de salinidad. Mantener buenas prácticas de fertilización.")

    # 8. Relación Ca:Mg
    if pd.notna(ca) and pd.notna(mg) and mg > 0:
        ratio = ca / mg
        if ratio > 8:
            recs.append(f"Relación Ca:Mg alta ({ratio:.1f}): Deficiencia inducida de Mg. Aplicar sulfato de magnesio (50-80 kg/ha) o cal dolomítica en próximo encalado.")
        elif ratio < 2:
            recs.append(f"Relación Ca:Mg baja ({ratio:.1f}): Posible exceso de Mg o deficiencia de Ca. Aplicar yeso agrícola (CaSO₄) para corregir el balance.")
        elif pd.notna(ca) and ca < 2.0:
            recs.append(f"Calcio bajo ({ca:.2f} meq/100g): Aplicar yeso agrícola (500-800 kg/ha) o cal. El Ca es esencial para la estructura celular y calidad de frutos.")

    # 9. Micronutrientes
    micro_issues = []
    if pd.notna(fe) and fe < 5:
        micro_issues.append(f"Fe bajo ({fe:.1f} ppm)")
    if pd.notna(zn) and zn < 1:
        micro_issues.append(f"Zn bajo ({zn:.1f} ppm)")
    if pd.notna(mn) and mn < 5:
        micro_issues.append(f"Mn bajo ({mn:.1f} ppm)")
    if pd.notna(b) and b < 0.2:
        micro_issues.append(f"B bajo ({b:.2f} ppm)")
    if micro_issues:
        recs.append(f"Deficiencias de micronutrientes detectadas: {', '.join(micro_issues)}. Aplicar fertilizante foliar multimineral o quelatos específicos.")
    elif pd.notna(zn) and zn > 20:
        warnings_list.append(f"Zinc excesivo ({zn:.1f} ppm): Puede ser fitotóxico. Evitar nuevas aplicaciones y monitorear síntomas foliares.")

    # 10. Sodio y estructura del suelo
    if pd.notna(na) and na > 1.0:
        warnings_list.append(f"Sodio elevado ({na:.2f} meq/100g): Riesgo de dispersión de arcillas y deterioro de estructura. Aplicar yeso agrícola (CaSO₄) para desplazar el Na⁺.")
    if pd.notna(da):
        if da > 1.4:
            recs.append(f"Densidad aparente alta ({da:.2f} g/cm³): Suelo compactado. Realizar subsolado o arado profundo (30-40 cm). Incorporar materia orgánica para mejorar porosidad.")
        elif da < 0.8:
            recs.append(f"Densidad aparente baja ({da:.2f} g/cm³): Suelo muy suelto. Buena condición para raíces, pero revisar retención hídrica y de nutrientes.")

    # 11. Textura del suelo
    if pd.notna(arena) and pd.notna(arcilla):
        if arena > 70:
            recs.append(f"Suelo arenoso ({arena:.0f}% arena): Baja retención de agua y nutrientes. Aumentar materia orgánica y fraccionar riegos y fertilizaciones.")
        elif arcilla > 50:
            recs.append(f"Suelo arcilloso ({arcilla:.0f}% arcilla): Posible encharcamiento y compactación. Mejorar drenaje, incorporar materia orgánica y evitar labranza en húmedo.")

    # 12. Recomendación específica por cultivo
    cultivo_recs = {
        'rosas': "Para rosas: pH óptimo 5.5–6.5, alta demanda de K y Ca. Fertilización fraccionada cada 15 días. Monitorear CE del sustrato (máx. 2 dS/m).",
        'tomate': "Para tomate: K elevado en fructificación (relación N:K = 1:1.5). Verificar Ca foliar para prevenir blossom-end rot. pH ideal 6.0–6.8.",
        'papa': "Para papa: pH óptimo 5.0–6.0. Controlar nemátodos con rotación de cultivos. Alta demanda de K y P en tuberización.",
        'maíz': "Para maíz: asegurar N suficiente en etapa V6 (60-80 kg N/ha). Monitorear Zn (deficiencia frecuente) y S. pH ideal 5.8–7.0.",
        'caña panelera': "Para caña panelera: alta demanda de K (150-200 kg K₂O/ha) y Mg. Aplicar silicio para resistencia a enfermedades y plagas.",
        'caña': "Para caña: fertilización fraccionada N-P-K. Silicio mejora la rigidez del tallo. Controlar salinidad en suelos con riego.",
        'aguacate': "Para aguacate: pH 5.5–7.0. Muy sensible al exceso de sales (CE < 1 dS/m) y al encharcamiento. Drenaje profundo crítico.",
        'mango': "Para mango: tolerante a suelos pobres pero responde bien a K y B en floración. Evitar exceso de N que promueve vegetación sobre fructificación.",
        'cacao': "Para cacao: pH óptimo 6.0–7.5. Alta demanda de K, Mg y B. Sombrío regulado mejora calidad del grano y salud del suelo.",
        'frijol': "Para frijol: inoculación con Rhizobium para fijación de N (puede reducir fertilización nitrogenada 50%). pH óptimo 6.0–7.0.",
        'fresas': "Para fresas: pH 5.5–6.5. Alta sensibilidad a salinidad (CE < 1 dS/m) y exceso de Na. Fertirrigación precisa recomendada.",
        'limón': "Para limón y cítricos: pH 5.5–6.5. Alta demanda de Mg y B. Aplicar Zn quelado si hay síntomas de clorosis intervenal.",
        'citricos': "Para cítricos: pH 5.5–6.5. Fertilizar con N fraccionado. Monitorear Mg, Fe y Zn. Sensibles a encharcamiento.",
        'aguacate hass': "Para aguacate Hass: suelos bien drenados, pH 6.0–7.0. Evitar compactación. Mulching orgánico para conservar humedad.",
        'platano': "Para plátano: alta demanda de K (200-250 kg K₂O/ha). pH 5.5–7.0. Monitorear Mg para evitar amarillamiento.",
        'banano': "Para banano: similar al plátano. K crítico. Aplicar Mg y Ca balanceados. Evitar suelos con mal drenaje.",
        'cebolla': "Para cebolla: pH 6.0–7.0. Alta demanda de S y Ca. Aplicar azufre para mejorar sabor y conservación.",
        'ajo': "Para ajo: pH 6.0–7.5. Requiere S para síntesis de alicina. Bulbificación favorecida con K alto y N bajo.",
        'maracuya': "Para maracuyá: pH 5.5–6.5. Alta demanda de K en fructificación. B importante para cuajado de frutos.",
        'piña': "Para piña: pH 4.5–6.0. Tolera suelos ácidos. Aplicar Fe quelado si hay clorosis. K elevado para calidad de fruto.",
        'arandano': "Para arándano: pH muy ácido requerido (4.5–5.5). Usar fertilizantes acidificantes (sulfato de amonio). Sensible a exceso de Ca.",
        'hortalizas': "Para hortalizas: pH 6.0–7.0. Alta demanda de N y Ca. Rotación de cultivos para manejo de enfermedades del suelo.",
        'pastos': "Para pastos: pH 5.5–6.5. Fertilización N fraccionada (urea o nitrato). S y Mg importantes para calidad nutritiva del pasto.",
        'pasturas': "Para pasturas: inoculación con micorrizas y bacterias fijadoras de N. Encalado periódico para mantener pH. Evitar sobrepastoreo.",
        'caucho': "Para caucho: pH 4.5–6.0. Tolera suelos ácidos. Fertilización moderada de N, P y K en establecimiento.",
        'cacao': "Para cacao: sombrío del 30-40% mejora productividad. K y Mg son críticos para llenado de mazorcas. pH 6.0–7.5.",
        'aromáticas': "Para plantas aromáticas: suelos bien drenados, pH 6.0–7.0. Bajo requerimiento de N (evitar exceso que reduce concentración de aceites esenciales).",
        'marañon': "Para marañón: tolera suelos pobres y ácidos (pH 5.0–6.5). Evitar suelos arcillosos mal drenados. Bajo requerimiento de fertilizantes.",
    }
    for key, val in cultivo_recs.items():
        if cultivo and key in str(cultivo).lower():
            recs.append(val)
            break

    # Garantizar mínimo 10 recomendaciones
    generales = [
        "Realizar análisis de suelo anualmente para monitorear evolución de nutrientes y ajustar el plan de fertilización.",
        "Implementar labranza mínima o cero labranza para conservar estructura del suelo y reducir erosión.",
        "Establecer coberturas vegetales o mulching para conservar humedad, regular temperatura y suprimir malezas.",
        "Inocular con micorrizas arbusculares en siembra para mejorar absorción de P y micronutrientes.",
        "Aplicar biofertilizantes (Azospirillum, Bacillus, Trichoderma) para mejorar disponibilidad de nutrientes y salud radicular.",
        "Rotar cultivos para romper ciclos de patógenos, mejorar estructura del suelo y diversificar la rizósfera.",
        "Gestionar los residuos de cosecha incorporándolos al suelo para aportar nutrientes y aumentar MO.",
        "Verificar la calidad del agua de riego (pH, CE, RAS) para evitar salinización o sodificación del suelo.",
        "Monitorear presencia de nemátodos y hongos de suelo mediante análisis microbiológico semestral.",
        "Diseñar un plan de fertilización integrado que combine fertilizantes minerales, orgánicos y biológicos.",
    ]

    total = len(recs) + len(warnings_list)
    if total < 10:
        needed = 10 - total
        for g in generales:
            if g not in recs:
                recs.append(g)
                needed -= 1
                if needed <= 0:
                    break

    return {'recomendaciones': recs, 'alertas': warnings_list}

def get_ai_recommendations(row_dict, use_api=False):
    """Genera recomendaciones. Si use_api=True y hay API key, usa Claude."""
    local = generate_local_recommendations(row_dict)
    if not use_api or not ANTHROPIC_API_KEY:
        return local

    try:
        ph = row_dict.get('pH', 'N/D')
        mo = row_dict.get('MO_pct', 'N/D')
        p = row_dict.get('P_disponible_ppm', 'N/D')
        cultivo = row_dict.get('cultivo', 'No especificado')
        prompt = f"""Eres un agrónomo experto. Analiza estos datos de suelo y da 3 recomendaciones concretas en español:
        pH: {ph}, Materia Orgánica: {mo}%, Fósforo disponible: {p} ppm, Cultivo: {cultivo}.
        Estado clasificado: {row_dict.get('estado','N/D')}
        Responde SOLO con JSON: {{"recomendaciones": ["rec1","rec2","rec3"], "alertas": ["alerta1"]}}"""

        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 500,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=10)
        data = resp.json()
        text = data['content'][0]['text']
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return local
    
# ── ML: Predicción por Cultivo + Municipio ──────────────────────────────────
CULTIVO_MUN_FEATURES = ['pH', 'MO_pct', 'P_disponible_ppm', 'Ca_meq_100g',
                         'Mg_meq_100g', 'K_meq_100g', 'Al_meq_100g', 'CE_dS_m']

le_cultivo   = LabelEncoder()
le_municipio = LabelEncoder()
clf_cm       = None
cm_acc       = 0.0

def train_cultivo_municipio_model(df):
    dft = df[['cultivo','municipio','estado'] + CULTIVO_MUN_FEATURES].dropna()
    dft = dft[dft['estado'].isin(['Óptimo','Medio','Crítico'])]
    dft = dft[dft['cultivo'].notna() & dft['municipio'].notna()]
    if len(dft) < 50:
        return None, None, None, 0.0

    le_c = LabelEncoder()
    le_m = LabelEncoder()
    dft = dft.copy()
    dft['cultivo_enc']   = le_c.fit_transform(dft['cultivo'])
    dft['municipio_enc'] = le_m.fit_transform(dft['municipio'])

    features = ['cultivo_enc','municipio_enc'] + CULTIVO_MUN_FEATURES
    X = dft[features]
    le_y = LabelEncoder()
    y = le_y.fit_transform(dft['estado'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, le_c, le_m, le_y, acc

try:
    result_cm = train_cultivo_municipio_model(df_global)
    if result_cm and len(result_cm) == 5:
        clf_cm, le_cultivo, le_municipio, le_estado_cm, cm_acc = result_cm
except Exception as e:
    print(f"Modelo cultivo-municipio omitido: {e}")
    le_estado_cm = None
# ── Helpers ─────────────────────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 3)
    except:
        return None

def row_to_dict(row):
    d = {}
    for k, v in row.items():
        if isinstance(v, (pd.Timestamp,)):
            d[k] = str(v.date()) if pd.notna(v) else None
        elif isinstance(v, float):
            d[k] = safe_float(v)
        elif pd.isna(v) if not isinstance(v, str) else False:
            d[k] = None
        else:
            d[k] = v
    return d

# ── Rutas ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/filters')
def get_filters():
    cultivos = sorted(df_global['cultivo'].dropna().unique().tolist())
    municipios = sorted(df_global['municipio'].dropna().unique().tolist())
    tipos_suelo = sorted(df_global['tipo_suelo'].dropna().unique().tolist())
    return jsonify({
        'cultivos': cultivos,
        'municipios': municipios,
        'tipos_suelo': tipos_suelo,
        'ph_range': [safe_float(df_global['pH'].min()), safe_float(df_global['pH'].max())],
        'mo_range': [safe_float(df_global['MO_pct'].min()), safe_float(df_global['MO_pct'].max())],
        'estados': ['Óptimo', 'Medio', 'Crítico']
    })

@app.route('/api/data')
def get_data():
    dff = df_global.copy()
    # Filters
    cultivo = request.args.get('cultivo')
    municipio = request.args.get('municipio')
    tipo_suelo = request.args.get('tipo_suelo')
    estado = request.args.get('estado')
    ph_min = request.args.get('ph_min', type=float)
    ph_max = request.args.get('ph_max', type=float)
    mo_min = request.args.get('mo_min', type=float)
    mo_max = request.args.get('mo_max', type=float)
    search = request.args.get('search', '').strip().lower()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    if cultivo and cultivo != 'all':
        dff = dff[dff['cultivo'] == cultivo]
    if municipio and municipio != 'all':
        dff = dff[dff['municipio'] == municipio]
    if tipo_suelo and tipo_suelo != 'all':
        dff = dff[dff['tipo_suelo'] == tipo_suelo]
    if estado and estado != 'all':
        dff = dff[dff['estado'] == estado]
    if ph_min is not None:
        dff = dff[dff['pH'] >= ph_min]
    if ph_max is not None:
        dff = dff[dff['pH'] <= ph_max]
    if mo_min is not None:
        dff = dff[dff['MO_pct'] >= mo_min]
    if mo_max is not None:
        dff = dff[dff['MO_pct'] <= mo_max]
    if search:
        mask = (dff['cultivo'].astype(str).str.lower().str.contains(search, na=False) |
                dff['municipio'].astype(str).str.lower().str.contains(search, na=False) |
                dff['cod_muestra'].astype(str).str.lower().str.contains(search, na=False))
        dff = dff[mask]

    total = len(dff)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = dff.iloc[start:end]
    records = [row_to_dict(r) for _, r in page_data.iterrows()]
    return jsonify({'records': records, 'total': total, 'page': page, 'per_page': per_page,
                    'pages': math.ceil(total / per_page)})

@app.route('/api/dashboard')
def get_dashboard():
    dff = df_global.copy()
    # Estado counts
    estado_counts = dff['estado'].value_counts().to_dict()
    # Cultivo distribution (top 12)
    cult_counts = dff['cultivo'].value_counts().head(12).to_dict()
    # pH distribution binned
    ph_bins = pd.cut(dff['pH'].dropna(), bins=[0,4,4.5,5,5.5,6,6.5,7,7.5,8,14],
                     labels=['<4','4-4.5','4.5-5','5-5.5','5.5-6','6-6.5','6.5-7','7-7.5','7.5-8','>8'])
    ph_dist = ph_bins.value_counts().sort_index().to_dict()
    # MO distribution
    mo_bins = pd.cut(dff['MO_pct'].dropna(), bins=[0,1,2,3,5,8,100],
                     labels=['<1%','1-2%','2-3%','3-5%','5-8%','>8%'])
    mo_dist = mo_bins.value_counts().sort_index().to_dict()
    # Texture
    textura = dff['tipo_suelo'].value_counts().head(8).to_dict()
    # Trend by year
    year_counts = dff[dff['año'].between(2018, 2026)].groupby('año').size().to_dict()
    # Stats
    stats_cols = ['pH','MO_pct','P_disponible_ppm','Ca_meq_100g','Mg_meq_100g','K_meq_100g']
    stats = {}
    for c in stats_cols:
        if c in dff.columns:
            col = dff[c].dropna()
            stats[c] = {
                'mean': safe_float(col.mean()), 'median': safe_float(col.median()),
                'std': safe_float(col.std()), 'min': safe_float(col.min()), 'max': safe_float(col.max()),
                'q25': safe_float(col.quantile(0.25)), 'q75': safe_float(col.quantile(0.75))
            }
    # Municipios top
    mun_counts = dff['municipio'].value_counts().head(10).to_dict()
    return jsonify({
        'estado_counts': estado_counts,
        'cultivo_counts': cult_counts,
        'ph_dist': ph_dist,
        'mo_dist': mo_dist,
        'textura': textura,
        'year_trend': year_counts,
        'stats': stats,
        'municipio_counts': mun_counts,
        'total_records': len(dff),
        'model_accuracy': round(model_acc * 100, 1) if model_acc else None
    })

@app.route('/api/record/<cod>')
def get_record(cod):
    rows = df_global[df_global['cod_muestra'] == cod]
    if rows.empty:
        return jsonify({'error': 'Not found'}), 404
    row = rows.iloc[0]
    d = row_to_dict(row)
    d['ai_recommendations'] = get_ai_recommendations(d, use_api=False)
    return jsonify(d)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    recs = get_ai_recommendations(data, use_api=bool(ANTHROPIC_API_KEY))
    return jsonify(recs)

@app.route('/api/predict', methods=['POST'])
def predict():
    if clf_model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    data = request.json
    try:
        X = pd.DataFrame([{f: data.get(f) for f in ML_FEATURES}])
        X = X.apply(pd.to_numeric, errors='coerce')
        missing = X.isnull().sum().sum()
        if missing > len(ML_FEATURES) // 2:
            return jsonify({'error': 'Datos insuficientes para predicción'})
        X = X.fillna(X.median() if not X.empty else 0)
        pred = clf_model.predict(X)[0]
        proba = clf_model.predict_proba(X)[0]
        classes = label_enc.classes_
        return jsonify({
            'prediction': label_enc.inverse_transform([pred])[0],
            'probabilities': {c: round(float(p)*100, 1) for c, p in zip(classes, proba)},
            'model_accuracy': round(model_acc * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/export')
def export():
    import io
    dff = df_global.copy()
    # Apply same filters
    for param in ['cultivo','municipio','estado']:
        val = request.args.get(param)
        if val and val != 'all':
            dff = dff[dff[param] == val]
    output = io.StringIO()
    dff.to_csv(output, index=False, sep=';', encoding='utf-8')
    from flask import Response
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=agroeco_export.csv'})
@app.route('/api/filter-recommendation')
def filter_recommendation():
    """
    Calcula promedios, clasificación dominante, tipo de suelo y genera
    recomendaciones agronómicas para el subconjunto filtrado de muestras.
    Incluye textura promedio para el triángulo USDA del frontend.
    """
    dff = df_global.copy()

    # Aplicar los mismos filtros que /api/data
    cultivo    = request.args.get('cultivo')
    municipio  = request.args.get('municipio')
    tipo_suelo = request.args.get('tipo_suelo')
    estado     = request.args.get('estado')
    ph_min     = request.args.get('ph_min', type=float)
    ph_max     = request.args.get('ph_max', type=float)
    mo_min     = request.args.get('mo_min', type=float)
    mo_max     = request.args.get('mo_max', type=float)

    if cultivo    and cultivo    != 'all': dff = dff[dff['cultivo']    == cultivo]
    if municipio  and municipio  != 'all': dff = dff[dff['municipio']  == municipio]
    if tipo_suelo and tipo_suelo != 'all': dff = dff[dff['tipo_suelo'] == tipo_suelo]
    if estado     and estado     != 'all': dff = dff[dff['estado']     == estado]
    if ph_min is not None: dff = dff[dff['pH'] >= ph_min]
    if ph_max is not None: dff = dff[dff['pH'] <= ph_max]
    if mo_min is not None: dff = dff[dff['MO_pct'] >= mo_min]
    if mo_max is not None: dff = dff[dff['MO_pct'] <= mo_max]

    if dff.empty:
        return jsonify({'error': 'No hay muestras con esos filtros.'}), 400

    # Columnas para promediar
    num_cols = [
        'pH', 'MO_pct', 'P_disponible_ppm', 'Ca_meq_100g', 'Mg_meq_100g',
        'K_meq_100g', 'Al_meq_100g', 'CE_dS_m', 'N_total_pct', 'C_organico_pct',
        'Fe_ppm', 'Zn_ppm', 'Mn_ppm', 'B_ppm', 'Na_meq_100g', 'densidad_aparente',
        'arena_pct', 'limo_pct', 'arcilla_pct', 'acidez_interc_meq',
        'CICE_meq_100g', 'S_ppm', 'Cu_ppm',
    ]
    promedios = {}
    for c in num_cols:
        if c in dff.columns:
            v = dff[c].dropna().mean()
            promedios[c] = safe_float(v)

    # Tipo de suelo predominante (moda) — prioriza tipo_suelo existente,
    # si no lo hay calcula desde textura promedio
    tipo_suelo_pred = None
    if 'tipo_suelo' in dff.columns:
        moda_ts = dff['tipo_suelo'].dropna().mode()
        if len(moda_ts) > 0:
            tipo_suelo_pred = moda_ts[0]
    if tipo_suelo_pred is None:
        tipo_suelo_pred = clasificar_textura(
            promedios.get('arena_pct'),
            promedios.get('limo_pct'),
            promedios.get('arcilla_pct')
        )

    # Cultivo más frecuente en la selección
    cultivo_moda = None
    if 'cultivo' in dff.columns:
        moda_c = dff['cultivo'].dropna().mode()
        cultivo_moda = moda_c[0] if len(moda_c) > 0 else None
    # Si el usuario filtró por cultivo, ese tiene precedencia
    cultivo_para_recs = (cultivo if cultivo and cultivo != 'all' else cultivo_moda) or ''

    # Construir fila promedio para recomendaciones
    avg_row = dict(promedios)
    avg_row['cultivo'] = cultivo_para_recs

    # Estado dominante
    estado_counts = dff['estado'].value_counts().to_dict()

    # Generar recomendaciones basadas en el promedio de la selección
    recs = generate_local_recommendations(avg_row)

    return jsonify({
        'total':                   len(dff),
        'estado_counts':           estado_counts,
        'promedios':               promedios,
        'tipo_suelo_predominante': tipo_suelo_pred,
        'cultivo_moda':            cultivo_moda,
        'recomendaciones':         recs['recomendaciones'],
        'alertas':                 recs['alertas'],
        'filtros_activos': {
            'cultivo':    cultivo    if cultivo    and cultivo    != 'all' else None,
            'municipio':  municipio  if municipio  and municipio  != 'all' else None,
            'tipo_suelo': tipo_suelo if tipo_suelo and tipo_suelo != 'all' else None,
            'estado':     estado     if estado     and estado     != 'all' else None,
        }
    })


if __name__ == '__main__':
    print("🌱 AgroEco Lab iniciando en http://localhost:5000")
    print(f"   Dataset: {len(df_global)} muestras cargadas")
    print(f"   Modelo ML: {'listo' if clf_model else 'no disponible'} (acc: {model_acc*100:.1f}%)" if clf_model else "   Modelo ML: no disponible")
    app.run(debug=True, host='0.0.0.0', port=5000)
