import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def extract_tract_from_geoid(geoid):
    """Extract 11-digit tract code from GEO_ID"""
    if pd.isna(geoid):
        return None
    geoid_str = str(geoid)
    match = re.search(r'(\d{11})$', geoid_str)
    return match.group(1) if match else geoid_str

def load_and_preprocess_data():
    """Load data"""
    stations = pd.read_csv('data/stations.csv')
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.lon, stations.lat),
        crs='EPSG:4326'
    )
    tracts_gdf = gpd.read_file('data/nyc_census_tracts_2020.geojson')
    income_data = pd.read_csv('data/ny_2020_income.csv')
    race_data = pd.read_csv('data/ny_2020_race.csv')

    return stations_gdf, tracts_gdf, income_data, race_data

def clean_and_merge_data(stations_gdf, tracts_gdf, income_data, race_data):
    """Merge station data with census tracts and calculate demographics"""
    tracts_clean = tracts_gdf[['geoid', 'boroname', 'geometry']].copy()
    tracts_clean = tracts_clean.to_crs(epsg=2263)
    stations_clean = stations_gdf.to_crs(epsg=2263).copy()

    # Spatial join with buffer
    stations_buffered = stations_clean.copy()
    stations_buffered['geometry'] = stations_buffered.geometry.buffer(50)
    stations_with_tracts = gpd.sjoin(
        stations_buffered,
        tracts_clean,
        how='left',
        predicate='intersects'
    )

    # Count stations per tract
    stations_per_tract = stations_with_tracts.groupby('geoid').size().reset_index()
    stations_per_tract.columns = ['geoid', 'station_count']
    merged_data = tracts_clean.merge(stations_per_tract, on='geoid', how='left')
    merged_data['station_count'] = merged_data['station_count'].fillna(0)

    # Calculate area and station density
    merged_data['area_sq_km'] = merged_data.geometry.area / 1e6
    merged_data['station_density'] = merged_data['station_count'] / merged_data['area_sq_km']
    merged_data['station_density'] = merged_data['station_density'].fillna(0)

    # Process census data
    income_data['geoid'] = income_data['GEO_ID'].apply(extract_tract_from_geoid)
    race_data['geoid'] = race_data['GEO_ID'].apply(extract_tract_from_geoid)

    income_col = 'S1903_C03_001E'
    income_to_merge = income_data[['geoid', income_col]].copy()
    income_to_merge.columns = ['geoid', 'median_income']
    income_to_merge['median_income'] = pd.to_numeric(income_to_merge['median_income'], errors='coerce')
    income_to_merge['median_income'] = income_to_merge['median_income'].clip(lower=0, upper=1000000)
    merged_data = merged_data.merge(income_to_merge, on='geoid', how='left')

    race_to_merge = race_data[['geoid', 'B02001_001E', 'B02001_002E']].copy()
    race_to_merge.columns = ['geoid', 'total_population', 'white_alone']
    race_to_merge['total_population'] = pd.to_numeric(race_to_merge['total_population'], errors='coerce')
    race_to_merge['white_alone'] = pd.to_numeric(race_to_merge['white_alone'], errors='coerce')

    # Calculate minority percentage
    mask = race_to_merge['total_population'] > 0
    race_to_merge['pct_minority'] = 0
    race_to_merge.loc[mask, 'pct_minority'] = 100 * (
            1 - race_to_merge.loc[mask, 'white_alone'] / race_to_merge.loc[mask, 'total_population']
    )
    merged_data = merged_data.merge(race_to_merge[['geoid', 'pct_minority', 'total_population']],
                                    on='geoid', how='left')

    # Calculate population density
    merged_data['population_density'] = merged_data['total_population'] / merged_data['area_sq_km']
    merged_data['population_density'] = merged_data['population_density'].fillna(0)

    # Fill missing values
    for col in ['median_income', 'pct_minority']:
        if col in merged_data.columns:
            borough_vals = merged_data.groupby('boroname')[col].transform(
                lambda x: x.median() if col == 'median_income' else x.mean())
            merged_data[col] = merged_data[col].fillna(borough_vals)
            overall_val = merged_data[col].median() if col == 'median_income' else merged_data[col].mean()
            merged_data[col] = merged_data[col].fillna(overall_val)

    return merged_data

def calculate_gini(station_counts):
    """Calculate Gini coefficient"""
    sorted_counts = np.sort(station_counts)
    n = len(sorted_counts)
    cum_counts = np.cumsum(sorted_counts)
    if cum_counts[-1] > 0:
        return (n + 1 - 2 * np.sum(cum_counts) / cum_counts[-1]) / n
    return 0

def calculate_lorenz_curve(station_counts):
    """Calculate Lorenz curve coordinates"""
    sorted_counts = np.sort(station_counts)
    if np.sum(sorted_counts) == 0:
        return np.linspace(0, 1, len(sorted_counts) + 1), np.linspace(0, 1, len(sorted_counts) + 1)
    n = len(sorted_counts)
    # X-axis: Cumulative % of tracts
    cum_population = np.arange(1, n + 1) / n
    # Y-axis: Cumulative % of stations
    cum_stations = np.cumsum(sorted_counts) / np.sum(sorted_counts)

    return np.insert(cum_population, 0, 0), np.insert(cum_stations, 0, 0)


def analyze_allocation_stats(data, station_col='station_count', name='Allocation'):
    """Analyze allocation statistics"""
    station_counts = data[station_col].values
    gini = calculate_gini(station_counts)
    tracts_with_stations = np.sum(station_counts > 0)
    coverage_rate = tracts_with_stations / len(station_counts)
    print(f"\n--- {name} Statistics ---")
    print(f"Gini Index: {gini:.3f}")
    print(f"Tracts with stations: {tracts_with_stations} ({coverage_rate:.1%})")

    if 'median_income' in data.columns and 'pct_minority' in data.columns:
        income_corr = np.corrcoef(data['median_income'], data[station_col])[0, 1]
        minority_corr = np.corrcoef(data['pct_minority'], data[station_col])[0, 1]
        print(f"Correlation with median income: {income_corr:.3f}")
        print(f"Correlation with minority: {minority_corr:.3f}")

    return gini

def calculate_deprivation_index(tracts_data):
    """Calculate deprivation index based on access, income, and race"""
    # Access Deprivation
    station_density = tracts_data['station_density'].values
    if np.max(station_density) > np.min(station_density):
        access_deprivation = 1 - (station_density - np.min(station_density)) / (
                np.max(station_density) - np.min(station_density))
    else:
        access_deprivation = np.ones_like(station_density) * 0.5

    # Income Deprivation
    income_deprivation = np.zeros_like(access_deprivation)
    if 'median_income' in tracts_data.columns:
        income = tracts_data['median_income'].values
        if np.max(income) > np.min(income):
            income_deprivation = 1 - (income - np.min(income)) / (np.max(income) - np.min(income))

    # Minority Deprivation
    minority_deprivation = np.zeros_like(access_deprivation)
    if 'pct_minority' in tracts_data.columns:
        minority_pct = tracts_data['pct_minority'].values / 100
        minority_deprivation = np.clip(minority_pct, 0, 1)

    # Set weights
    weights = [0.4, 0.3, 0.3]
    deprivation_index = weights[0] * access_deprivation + weights[1] * income_deprivation + weights[
        2] * minority_deprivation
    tracts_data['deprivation_index'] = deprivation_index

    # Ensure Normalization
    min_dep = np.min(deprivation_index)
    max_dep = np.max(deprivation_index)
    tracts_data['deprivation_index_norm'] = (deprivation_index - min_dep) / (
            max_dep - min_dep) if max_dep > min_dep else 0.5

    return tracts_data

def build_spatial_network(tracts_data, distance_threshold=2000):
    """Build network"""
    centroids = tracts_data.geometry.centroid
    coords = np.array([(c.x, c.y) for c in centroids])
    dist_matrix = cdist(coords, coords, metric='euclidean')
    adjacency_matrix = (dist_matrix < distance_threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    G = nx.Graph()
    for idx, row in tracts_data.iterrows():
        G.add_node(idx, deprivation_index_norm=row.get('deprivation_index_norm', 0))

    edges = np.where(adjacency_matrix > 0)
    for i, j in zip(edges[0], edges[1]):
        if i < j:
            weight = 1 / (dist_matrix[i, j] + 1)
            G.add_edge(i, j, weight=weight)
    return G

def detect_communities(G, tracts_data):
    """Community Detection"""
    import community as community_louvain
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    tracts_data['community'] = tracts_data.index.map(partition)

    return tracts_data, partition

def equity_aware_allocation(tracts_data, total_stations, equity_weight=0.5):
    """Allocate stations based on Equity and Efficiency"""
    # Efficiency Score
    efficiency_metric = tracts_data['population_density']
    eff_min, eff_max = efficiency_metric.min(), efficiency_metric.max()
    efficiency_score = (efficiency_metric - eff_min) / (eff_max - eff_min) if eff_max > eff_min else 0.5

    # Equity Score
    equity_score = tracts_data['deprivation_index_norm']

    # Allocation Score
    combined_score = equity_weight * equity_score + (1 - equity_weight) * efficiency_score
    tracts_data['allocation_score'] = combined_score

    # Allocate stations
    tracts_sorted = tracts_data.sort_values('allocation_score', ascending=False).reset_index(drop=True)
    tracts_sorted['allocated_stations'] = 0

    community_weights = tracts_sorted.groupby('community')['allocation_score'].sum()
    community_weights = community_weights / community_weights.sum()

    for community, weight in community_weights.items():
        community_stations = int(np.round(weight * total_stations))
        community_tracts = tracts_sorted[tracts_sorted['community'] == community]

        if len(community_tracts) > 0:
            # Ensure top tract gets at least 1
            top_tract_idx = community_tracts.index[0]
            tracts_sorted.loc[top_tract_idx, 'allocated_stations'] = 1
            community_stations -= 1

            # Proportional distribution of remainder
            if community_stations > 0:
                comm_scores = community_tracts['allocation_score'].sum()
                for idx in community_tracts.index:
                    tract_share = tracts_sorted.loc[idx, 'allocation_score'] / comm_scores
                    additional = int(np.floor(community_stations * tract_share))
                    tracts_sorted.loc[idx, 'allocated_stations'] += additional

    return tracts_sorted

def generate_visualizations(current_data, equity_data, extreme_equity_data, extreme_efficiency_data):
    gini_current = calculate_gini(current_data['station_count'].values)
    gini_equity = calculate_gini(equity_data['allocated_stations'].values)
    curr_x, curr_y = calculate_lorenz_curve(current_data['station_count'].values)
    eq_x, eq_y = calculate_lorenz_curve(equity_data['allocated_stations'].values)

    if 'community' in current_data.columns:
        community_boundaries = current_data.dissolve(by='community')
    else:
        community_boundaries = None

    # Fig 1: Lorenz Curve & Gini of Current NYC
    plt.figure(figsize=(8, 6))
    plt.plot(curr_x, curr_y, 'b-', linewidth=2, label=f'Current Allocation (Gini = {gini_current:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Equality')
    plt.fill_between(curr_x, curr_x, curr_y, alpha=0.1, color='blue')

    plt.xlabel('Cumulative Percentage of Census Tracts')
    plt.ylabel('Cumulative Percentage of Bike Stations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/fig1_current_lorenz.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Fig 2: Lorenz Comparison (Current vs Equity-Aware)
    plt.figure(figsize=(8, 6))
    plt.plot(curr_x, curr_y, 'b-', linewidth=2, label=f'Current (Gini = {gini_current:.3f})')
    plt.plot(eq_x, eq_y, 'r-', linewidth=2, label=f'Equity-Aware (Gini = {gini_equity:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Equality')

    plt.xlabel('Cumulative Percentage of Census Tracts')
    plt.ylabel('Cumulative Percentage of Bike Stations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/fig2_lorenz_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Fig 3: Station Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot current distribution
    current_data.plot(column='station_count',
                      cmap='Blues',
                      linewidth=0.1,
                      edgecolor='grey',
                      legend=True,
                      ax=axes[0],
                      legend_kwds={'label': 'Station Count', 'shrink': 0.7})
    axes[0].set_title('Current Station Distribution', fontsize=14)
    axes[0].axis('off')

    # Plot equity-aware distribution
    equity_data.plot(column='allocated_stations',
                     cmap='Reds',
                     linewidth=0.1,
                     edgecolor='grey',
                     legend=True,
                     ax=axes[1],
                     legend_kwds={'label': 'Allocated Stations', 'shrink': 0.7})

    if community_boundaries is not None:
        community_boundaries.plot(ax=axes[1], facecolor='none', edgecolor='black', linewidth=1.5)

    axes[1].set_title('Equity-Aware Station Distribution\n(Equity Weight $\\alpha$ = 0.5)', fontsize=14)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('results/fig3_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Fig 4: Extreme Cases
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Complete Equity (Weight = 1)
    extreme_equity_data.plot(column='allocated_stations',
                             cmap='Purples',
                             linewidth=0.1,
                             edgecolor='grey',
                             legend=True,
                             ax=axes[0],
                             legend_kwds={'label': 'Allocated Stations', 'shrink': 0.7})
    axes[0].set_title('Scenario A: Complete Equity\n(Equity Weight $\\alpha$ = 1.0)', fontsize=14)
    axes[0].axis('off')

    # Complete Efficiency (Weight = 0)
    extreme_efficiency_data.plot(column='allocated_stations',
                                 cmap='Greens',
                                 linewidth=0.1,
                                 edgecolor='grey',
                                 legend=True,
                                 ax=axes[1],
                                 legend_kwds={'label': 'Allocated Stations', 'shrink': 0.7})
    axes[1].set_title('Scenario B: Complete Efficiency\n(Equity Weight $\\alpha$ = 0.0)', fontsize=14)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('results/fig4_extreme_cases.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load data and preprocess
    stations_gdf, tracts_gdf, income_data, race_data = load_and_preprocess_data()
    merged_data = clean_and_merge_data(stations_gdf, tracts_gdf, income_data, race_data)

    # Current allocation
    analyze_allocation_stats(merged_data, station_col='station_count', name='Current Allocation')

    # Equity-aware algorithm
    merged_data = calculate_deprivation_index(merged_data)
    G = build_spatial_network(merged_data, distance_threshold=2000)
    merged_data, partition = detect_communities(G, merged_data)
    total_stations = int(merged_data['station_count'].sum())
    equity_data = equity_aware_allocation(
        merged_data.copy(), total_stations=total_stations, equity_weight=0.5
    )
    analyze_allocation_stats(equity_data, station_col='allocated_stations', name='Balanced Equity-Aware (Weight=0.5)')

    # Complete Equity (Weight=1.0)
    extreme_equity_data = equity_aware_allocation(
        merged_data.copy(), total_stations=total_stations, equity_weight=1.0
    )

    # Complete Efficiency (Weight=0.0)
    extreme_efficiency_data = equity_aware_allocation(
        merged_data.copy(), total_stations=total_stations, equity_weight=0.0
    )

    # Generate Figures
    generate_visualizations(
        merged_data,
        equity_data,
        extreme_equity_data,
        extreme_efficiency_data
    )

    print("\nFigures saved in 'results/' folder.")


if __name__ == "__main__":
    main()