#!/usr/bin/env python3
"""
Klassificering av uppvärmningsmetoder för ny temperaturdata.
Automatisk laddning av alla .xls/.xlsx-filer, med datumfilter från 1 juni.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import glob
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# KATEGORIER (Samma som original)
# ============================================================================
HEATING_CATEGORIES = {
    1: {'name': 'Konstant vädring', 'description': 'Följer med utomhustemperaturen - ständig vädring', 'color': '#1f77b4'},
    2: {'name': 'Nattvädring med solskydd', 'description': 'Följer utomhustemperaturen nattetid, fördröjning dagtid', 'color': '#ff7f0e'},
    3: {'name': 'Solinstrålning', 'description': 'Markant temperaturökning vissa dagar dagtid - solinstrålning', 'color': '#2ca02c'},
    4: {'name': 'Internlaster', 'description': 'Oregelbunden uppvärmning - majoriteten internlaster', 'color': '#d62728'},
    5: {'name': 'Värmesystem igång', 'description': 'Övertemperaturer vid låg utomhustemperatur (>21°C) nattetid', 'color': '#9467bd'},
}

# ============================================================================
# 1. LADDNING AV DATA (Automatisk + datumfilter)
# ============================================================================
def load_temperature_data(file_path):
    """Ladda data från textfil (specialformat) och filtrera datum >= 1 juni"""
    try:
        # Läs filen som text
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        # Hitta start av data (efter [#D])
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == '[#D]':
                data_start = i + 1
                break
        
        if data_start is None:
            print(f"❌ Fel: [#D] tagg hittades inte i {file_path}")
            return pd.DataFrame(columns=['timestamp', 'temperature'])
        
        # Läs data
        data_lines = lines[data_start:]
        rows = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                date_str = parts[0].strip()
                time_str = parts[1].strip()
                temp_str = parts[3].strip().replace(',', '.')
                try:
                    timestamp = pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
                    temperature = float(temp_str)
                    if pd.notna(timestamp):
                        rows.append({'timestamp': timestamp, 'temperature': temperature})
                except:
                    continue
        
        data_df = pd.DataFrame(rows)
        if len(data_df) == 0:
            print(f"⚠️  Varning: {file_path} har ingen data från/med 1 juni.")
            return pd.DataFrame(columns=['timestamp', 'temperature'])
        
        # ⬇️ ⬇️ FILTRERA DATUM: Endast data från och med 1 juni ⬇️ ⬇️
        data_df = data_df[data_df['timestamp'].dt.date >= pd.to_datetime('2024-06-01').date()]
        
        if len(data_df) == 0:
            print(f"⚠️  Varning: {file_path} har ingen data från/med 1 juni.")
        
        return data_df[['timestamp', 'temperature']]
    except Exception as e:
        print(f"❌ Fel vid laddning av {file_path}: {e}")
        return pd.DataFrame(columns=['timestamp', 'temperature'])

def load_all_data():
    """Ladda alla .xls/.xlsx-filer automatiskt"""
    # Hitta alla Excel-filer
    excel_files = glob.glob('*.xls') + glob.glob('*.xlsx')
    if not excel_files:
        print("❌ Fel: Inga .xls/.xlsx-filer hittades i mappen.")
        return {}, None

    # Separera indoor och outdoor
    indoor_data = {}
    outdoor_df = None

    for file_path in excel_files:
        # 61_61.xls är utomhus
        if '61_61' in file_path:
            outdoor_df = load_temperature_data(file_path)
            if outdoor_df is not None and len(outdoor_df) > 0:
                outdoor_df.columns = ['timestamp', 'outdoor_temp']
                print(f"✅ Laddat utomhus (61_61.xls): {len(outdoor_df)} mätningar, intervall: {outdoor_df['outdoor_temp'].min():.1f}-{outdoor_df['outdoor_temp'].max():.1f}°C")
        else:
            df = load_temperature_data(file_path)
            if df is not None and len(df) > 0:
                # Använd filnamnet som sensor-ID (utan ändelse)
                sensor_id = file_path.split('.')[0]
                indoor_data[sensor_id] = df
                print(f"✅ Laddat {sensor_id}: {len(df)} mätningar, intervall: {df['temperature'].min():.1f}-{df['temperature'].max():.1f}°C")

    if not indoor_data:
        print("❌ Fel: Inga indoor-filer hittades.")
        return {}, None
    if outdoor_df is None or len(outdoor_df) == 0:
        print("❌ Fel: Ingen utomhusdata (61_61.xls) hittades.")
        return {}, None

    return indoor_data, outdoor_df

# ============================================================================
# 2. EXTRAHERA NYCKELFUNKTIONER
# ============================================================================
def extract_features_for_clustering(indoor_df, outdoor_df):
    merged = pd.merge(indoor_df, outdoor_df, on='timestamp', how='inner')
    if len(merged) == 0:
        print("⚠️  Varning: Inga matchande tidsstämplar mellan indoor och outdoor.")
        return None

    merged['hour'] = merged['timestamp'].dt.hour
    merged['is_day'] = merged['hour'].between(6, 18).astype(int)
    merged['is_night'] = (~merged['hour'].between(6, 18)).astype(int)
    merged['temp_diff'] = merged['temperature'] - merged['outdoor_temp']

    daily = merged.groupby(merged['timestamp'].dt.date)
    daily_max = daily['temperature'].max()
    daily_min = daily['temperature'].min()
    daily_range = daily_max - daily_min

    daily_night_min = merged[merged['is_night'] == 1].groupby(merged['timestamp'].dt.date)['temperature'].min()
    daily_spike = daily_max - daily_night_min

    features = {
        'mean_temp': merged['temperature'].mean(),
        'std_temp': merged['temperature'].std(),
        'mean_diff': merged['temp_diff'].mean(),
        'corr_outdoor': merged['temperature'].corr(merged['outdoor_temp']),
        'corr_day': merged[merged['is_day'] == 1]['temperature'].corr(merged[merged['is_day'] == 1]['outdoor_temp']) if len(merged[merged['is_day'] == 1]) > 0 else 0,
        'corr_night': merged[merged['is_night'] == 1]['temperature'].corr(merged[merged['is_night'] == 1]['outdoor_temp']) if len(merged[merged['is_night'] == 1]) > 0 else 0,
        'mean_daily_spike': daily_spike.mean(),
        'overheating_low_outdoor': (merged[merged['outdoor_temp'] < 10]['temperature'] > 21).mean() if len(merged[merged['outdoor_temp'] < 10]) > 0 else 0,
        'mean_night_temp': merged[merged['is_night'] == 1]['temperature'].mean() if len(merged[merged['is_night'] == 1]) > 0 else 0,
    }
    return features

def create_feature_matrix(indoor_data, outdoor_df):
    sensor_names = sorted(indoor_data.keys())
    feature_list = []
    for sensor_id in sensor_names:
        features = extract_features_for_clustering(indoor_data[sensor_id], outdoor_df)
        if features is not None:
            feature_list.append(features)
        else:
            print(f"⚠️  Varning: {sensor_id} har ingen matchande data med utomhus.")
    return pd.DataFrame(feature_list, index=sensor_names), sensor_names

# ============================================================================
# 3. MANUELL KLASSIFICERING
# ============================================================================
def manual_classification(df_features, sensor_names):
    classifications = {}
    for sensor_id in sensor_names:
        features = df_features.loc[sensor_id]
        if features['overheating_low_outdoor'] > 0.5 and features['mean_night_temp'] > 22:
            category = 5
        elif features['mean_daily_spike'] > 2.5 and features['mean_night_temp'] > 21:
            category = 3
        elif features['corr_night'] > 0.6 and features['corr_day'] < 0.5:
            category = 2
        elif features['corr_outdoor'] > 0.7 and abs(features['mean_diff']) < 3:
            category = 1
        elif features['corr_outdoor'] < 0.4 and features['std_temp'] > 2.0:
            category = 4
        elif features['mean_diff'] > 6 and features['overheating_low_outdoor'] > 0.3:
            category = 5
        elif features['mean_daily_spike'] > 1.5:
            category = 3
        else:
            category = 5
        classifications[sensor_id] = category
    return classifications

# ============================================================================
# 4. VISUALISERING
# ============================================================================
def plot_results(classifications, df_features, sensor_names, indoor_data, outdoor_df):
    # --- 1. Klassificeringssammanfattning ---
    plt.figure(figsize=(14, 8))
    colors = [HEATING_CATEGORIES[classifications[s]]['color'] for s in sensor_names]
    for i, sensor in enumerate(sensor_names):
        cat = classifications[sensor]
        plt.bar(i, cat, color=colors[i], alpha=0.8, width=0.6)
    plt.xticks(range(len(sensor_names)), sensor_names, rotation=45)
    plt.yticks(range(1, 6), [HEATING_CATEGORIES[i]['name'] for i in range(1, 6)])
    plt.title('Lägenhetsklassificering - MANUELL METOD')
    plt.ylabel('Kategori')
    plt.tight_layout()
    plt.savefig('heating_classification_manual_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Sparat: heating_classification_manual_summary.png")

    # --- 2. Feature-heatmap ---
    df_features_copy = df_features.copy()
    df_features_copy['category'] = [classifications[s] for s in sensor_names]
    df_norm = df_features_copy[['corr_outdoor', 'corr_day', 'corr_night', 'mean_diff', 'mean_daily_spike', 'overheating_low_outdoor', 'mean_night_temp']].copy()
    scaler = MinMaxScaler()
    df_norm[df_norm.columns] = scaler.fit_transform(df_norm)
    df_norm['category'] = df_features_copy['category']
    df_norm = df_norm.sort_values('category')
    df_norm = df_norm.drop(columns=['category'])

    plt.figure(figsize=(14, 8))
    sns.heatmap(df_norm.T, cmap='viridis', annot=True, fmt='.2f',
                yticklabels=['Korr utomhus', 'Korr dag', 'Korr natt', 'Medel diff', 'Daglig spike', 'Övertemp låg utomhus', 'Medel nattemp'])
    plt.title('Normaliserade egenskaper per lägenhet (sorterad efter kategori)')
    plt.tight_layout()
    plt.savefig('feature_heatmap_manual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Sparat: feature_heatmap_manual.png")

    # --- 3. Korrelationsjämförelse ---
    plt.figure(figsize=(14, 6))
    x = np.arange(len(sensor_names))
    width = 0.35
    corr_day = df_features['corr_day'].fillna(0)
    corr_night = df_features['corr_night'].fillna(0)
    corr_outdoor = df_features['corr_outdoor'].fillna(0)
    plt.bar(x - width, corr_outdoor, width, label='Total korrelation', color='#1f77b4')
    plt.bar(x, corr_day, width, label='Dagkorrelation', color='#ff7f0e')
    plt.bar(x + width, corr_night, width, label='Nattkorrelation', color='#2ca02c')
    plt.xlabel('Lägenhet')
    plt.ylabel('Korrelation')
    plt.title('Korrelation med utomhustemperatur (Total, Dag, Natt)')
    plt.xticks(x, sensor_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('correlation_comparison_manual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Sparat: correlation_comparison_manual.png")

    # --- 4. PCA-klustring ---
    cluster_features = ['corr_outdoor', 'corr_day', 'corr_night', 'mean_diff', 'mean_daily_spike', 'overheating_low_outdoor', 'mean_night_temp']
    X = df_features[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 10))
    colors = [HEATING_CATEGORIES[classifications[s]]['color'] for s in sensor_names]
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=100)
    for i, sensor in enumerate(sensor_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], sensor, fontsize=9, ha='right', va='bottom')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA: Lägenheter i egenskapsrum - MANUELL METOD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_clustering_manual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Sparat: pca_clustering_manual.png")

    # --- 5. Temperaturmönster per kategori ---
    for cat_id in range(1, 6):
        sensors_in_cat = [s for s in sensor_names if classifications[s] == cat_id]
        if sensors_in_cat:
            plt.figure(figsize=(14, 8))
            for sensor in sensors_in_cat:
                df = indoor_data[sensor]
                plt.plot(df['timestamp'], df['temperature'], label=sensor, alpha=0.7)
            plt.plot(outdoor_df['timestamp'], outdoor_df['outdoor_temp'], label='Utomhus', color='k', linewidth=2, linestyle='--')
            plt.xlabel('Datum')
            plt.ylabel('Temperatur (°C)')
            plt.title(f'Temperaturmönster - Kategori {cat_id}: {HEATING_CATEGORIES[cat_id]["name"]}')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'temperature_patterns_category_{cat_id}_manual.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Sparat: temperature_patterns_category_{cat_id}_manual.png")

    # --- 6. Medeltemperatur per timme för alla lägenheter ---
    plt.figure(figsize=(14, 8))
    colors = [HEATING_CATEGORIES[classifications[s]]['color'] for s in sensor_names]
    for i, sensor in enumerate(sensor_names):
        df = indoor_data[sensor]
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['temperature'].mean()
        plt.plot(range(24), hourly_avg, label=sensor, color=colors[i], marker='o', linewidth=2, markersize=4)
    
    # Lägg till utomhusdata
    outdoor_df['hour'] = outdoor_df['timestamp'].dt.hour
    outdoor_hourly_avg = outdoor_df.groupby('hour')['outdoor_temp'].mean()
    plt.plot(range(24), outdoor_hourly_avg, label='Utomhus', color='k', linewidth=2, linestyle='--', marker='s')
    
    plt.xlabel('Timme (0-24)')
    plt.ylabel('Medeltemperatur (°C)')
    plt.title('Medeltemperatur per timme under hela mätperioden')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hourly_average_temperature.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Sparat: hourly_average_temperature.png")

# ============================================================================
# 5. RAPPORTGENERERING
# ============================================================================
def generate_report(classifications, df_features, sensor_names):
    print("\n" + "="*80)
    print("UPPVÄRMNINGSMETODOLOGI RAPPORT - MANUELL METOD")
    print("="*80)

    categories_count = {}
    for cat in classifications.values():
        categories_count[cat] = categories_count.get(cat, 0) + 1

    print("\nKlassificering av lägenheter:")
    for cat_id, count in sorted(categories_count.items()):
        print(f"  Kategori {cat_id}: {HEATING_CATEGORIES[cat_id]['name']} - {count} lägenhet(er)")

    print("\n" + "="*80)
    print("DETALJERAD ANALYS PER LÄGENHET")
    print("="*80)

    for sensor_id in sensor_names:
        cat = classifications[sensor_id]
        features = df_features.loc[sensor_id]
        print(f"\n{sensor_id}:")
        print(f"  Kategori: {HEATING_CATEGORIES[cat]['name']}")
        print(f"  Beskrivning: {HEATING_CATEGORIES[cat]['description']}")
        print(f"  Nyckelvärden:")
        print(f"    Medeltemperatur: {features['mean_temp']:.2f}°C")
        print(f"    Standardavvikelse: {features['std_temp']:.2f}°C")
        print(f"    Korrelation utomhus: {features['corr_outdoor']:.3f}")
        print(f"    Dagkorrelation: {features['corr_day']:.3f}")
        print(f"    Nattkorrelation: {features['corr_night']:.3f}")
        print(f"    Medeldifferens: {features['mean_diff']:.2f}°C")
        print(f"    Medel daglig ökning: {features['mean_daily_spike']:.2f}°C")
        print(f"    Övertemperatur vid låg utomhus: {features['overheating_low_outdoor']*100:.1f}%")
        print(f"    Medelnattemperatur: {features['mean_night_temp']:.2f}°C")

    print("\n" + "="*80)
    print("SAMMANFATTNING OCH REKOMMENDATIONER")
    print("="*80)

    if 5 in categories_count:
        cat_sensors = [s for s in sensor_names if classifications[s] == 5]
        print(f"\n🔥 Kategori 5 - Värmesystem igång ({len(cat_sensors)} lägenheter): {cat_sensors}")
        print("   • Justera värmesystemet för att undvika övertemperatur")
        print("   • Överväg tidstyrning eller zonsindelning")

    if 3 in categories_count:
        cat_sensors = [s for s in sensor_names if classifications[s] == 3]
        print(f"\n☀️  Kategori 3 - Solinstrålning ({len(cat_sensors)} lägenheter): {cat_sensors}")
        print("   • Installera bättre solskydd (persienner, markiser)")
        print("   • Överväg reflektorer eller solfilm på fönster")

    if 2 in categories_count:
        cat_sensors = [s for s in sensor_names if classifications[s] == 2]
        print(f"\n🌙 Kategori 2 - Nattvädring med solskydd ({len(cat_sensors)} lägenheter): {cat_sensors}")
        print("   • Solskydd fungerar bra - behåll aktuell lösning")
        print("   • Överväg att öka ventilationen dagtid för bättre komfort")

    if 1 in categories_count:
        cat_sensors = [s for s in sensor_names if classifications[s] == 1]
        print(f"\n🌬️  Kategori 1 - Konstant vädring ({len(cat_sensors)} lägenheter): {cat_sensors}")
        print("   • Behåll nuvarande ventilationssystem")

    if 4 in categories_count:
        cat_sensors = [s for s in sensor_names if classifications[s] == 4]
        print(f"\n🏢 Kategori 4 - Internlaster ({len(cat_sensors)} lägenheter): {cat_sensors}")
        print("   • Optimal för kontor med många människor")

    print("\n" + "="*80)
    print("ANALYS SLUTFÖRD")
    print("="*80)

# ============================================================================
# 6. HUVUDFUNKTION
# ============================================================================
def main():
    print("="*80)
    print("UPPVÄRMNINGSMETODOLOGI - NY DATA (AUTOMATISK LADDNING)")
    print("="*80)

    # Ladda data
    print("\n📂 Laddar temperaturdata...")
    indoor_data, outdoor_df = load_all_data()

    if not indoor_data or outdoor_df is None:
        print("❌ Fel: Kunde inte ladda data. Kontrollera att .xls/.xlsx-filer finns i mappen.")
        return

    # Skapa feature-matris
    print("\n🔍 Extraherar egenskaper...")
    df_features, sensor_names = create_feature_matrix(indoor_data, outdoor_df)

    if len(df_features) == 0:
        print("❌ Fel: Ingen data kunde matchas mellan indoor och outdoor.")
        return

    # Klassificera
    print("\n🏷️  Klassificerar lägenheter...")
    classifications = manual_classification(df_features, sensor_names)

    # Generera diagram
    print("\n📊 Skapar diagram...")
    plot_results(classifications, df_features, sensor_names, indoor_data, outdoor_df)

    # Generera rapport
    print("\n📄 Genererar rapport...")
    generate_report(classifications, df_features, sensor_names)

    print("\n✅ KLAR! Alla diagram och rapport har sparats.")

if __name__ == "__main__":
    main()
