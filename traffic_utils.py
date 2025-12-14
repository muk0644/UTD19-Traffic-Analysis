"""
Urban Traffic Analysis Utilities

This module provides reusable utility functions for analyzing urban traffic data
from the UTD19 Multi-City Traffic Detector Dataset (ETH Zurich).

PURPOSE:
--------
- Code reusability: Extract common operations into importable functions
- Software engineering best practices: Modular, tested, documented code
- Portfolio demonstration: Shows professional Python development skills
- Future scalability: Easy to extend and test without modifying notebooks

OPTIONAL NOTE:
--------------
This module is OPTIONAL. The main analysis can be done entirely within
traffic_analysis.ipynb without using these utilities. However, using this
module demonstrates better software engineering practices and makes code
more maintainable and reusable for future projects.

Functions:
    - load_and_filter_detectors(): Load and filter detector metadata by city
    - find_nearest_detectors(): Find N nearest detectors using Euclidean distance
    - compute_detector_statistics(): Calculate mean occupancy and flow statistics
    - apply_kmeans_clustering(): Apply K-Means clustering with configuration
    - elbow_method(): Determine optimal number of clusters
    - plot_detector_map(): Create interactive Folium map visualization
    - plot_clusters(): Visualize K-Means clustering results
"""

import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_and_filter_detectors(detectors_path, city_code='london'):
    """
    Load detector metadata and filter by city code.
    
    Parameters:
    -----------
    detectors_path : str
        Path to detectors_public.csv file
    city_code : str
        City code to filter (default: 'london')
    
    Returns:
    --------
    pd.DataFrame
        Filtered detector dataframe with columns:
        - detid: Detector ID
        - lat: Latitude
        - long: Longitude
        - citycode: City code
        - road_type: Road type
    
    Example:
    --------
    >>> detectors = load_and_filter_detectors('data/detectors_public.csv', 'london')
    >>> print(f"Found {len(detectors)} London detectors")
    """
    detectors = pd.read_csv(detectors_path)
    detectors_filtered = detectors[detectors['citycode'] == city_code].reset_index(drop=True)
    return detectors_filtered


def find_nearest_detectors(detectors_df, target_detid, n_neighbors=5):
    """
    Find N nearest detectors to a target detector using Euclidean distance.
    
    Parameters:
    -----------
    detectors_df : pd.DataFrame
        DataFrame with detector locations (lat, long columns)
    target_detid : str
        Target detector ID to find neighbors for
    n_neighbors : int
        Number of nearest neighbors to return (default: 5)
    
    Returns:
    --------
    list
        List of N nearest detector IDs
    
    Example:
    --------
    >>> neighbors = find_nearest_detectors(detectors, 'CNTR_N03/164a1', 5)
    >>> print(f"5 nearest detectors: {neighbors}")
    """
    target = detectors_df[detectors_df['detid'] == target_detid]
    if len(target) == 0:
        raise ValueError(f"Detector {target_detid} not found")
    
    target_lat = target['lat'].values[0]
    target_long = target['long'].values[0]
    
    # Calculate Euclidean distance to all detectors
    distances = {}
    for idx, row in detectors_df.iterrows():
        if row['detid'] != target_detid:
            dist = np.sqrt((row['long'] - target_long)**2 + (row['lat'] - target_lat)**2)
            distances[row['detid']] = dist
    
    # Sort and return N nearest
    sorted_detectors = sorted(distances.items(), key=lambda x: x[1])
    nearest = [det_id for det_id, _ in sorted_detectors[:n_neighbors]]
    return nearest


def compute_detector_statistics(data_df, detectors_df, n_detectors=100):
    """
    Compute mean occupancy and flow for detectors.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Traffic measurements with columns: detid, occ, flow
    detectors_df : pd.DataFrame
        Detector metadata with location information
    n_detectors : int
        Number of detectors to compute statistics for (default: 100)
    
    Returns:
    --------
    pd.DataFrame
        Statistics dataframe with columns: mean_occ, mean_flow
    
    Example:
    --------
    >>> stats = compute_detector_statistics(data, detectors, 100)
    >>> print(stats.head())
    """
    stats_list = []
    
    for idx, detector in detectors_df.head(n_detectors).iterrows():
        detector_data = data_df[data_df['detid'] == detector['detid']]
        
        mean_occ = detector_data['occ'].mean()
        mean_flow = detector_data['flow'].mean()
        
        stats_list.append({
            'mean_occ': mean_occ,
            'mean_flow': mean_flow
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def apply_kmeans_clustering(data_df, n_clusters=7, random_state=0):
    """
    Apply K-Means clustering to occupancy and flow data.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame with 'mean_occ' and 'mean_flow' columns
    n_clusters : int
        Number of clusters (default: 7)
    random_state : int
        Random seed for reproducibility (default: 0)
    
    Returns:
    --------
    tuple
        (kmeans_model, cluster_labels, centroids)
    
    Example:
    --------
    >>> kmeans, labels, centroids = apply_kmeans_clustering(stats, 7)
    >>> print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    """
    X = data_df[['mean_occ', 'mean_flow']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    return kmeans, labels, kmeans.cluster_centers_


def elbow_method(data_df, k_range=range(1, 10)):
    """
    Perform elbow method to find optimal number of clusters.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame with 'mean_occ' and 'mean_flow' columns
    k_range : range
        Range of cluster numbers to test (default: 1-10)
    
    Returns:
    --------
    dict
        Dictionary with k values as keys and SSE values as values
    
    Example:
    --------
    >>> sse_dict = elbow_method(stats)
    >>> plt.plot(list(sse_dict.keys()), list(sse_dict.values()))
    >>> plt.show()
    """
    X = data_df[['mean_occ', 'mean_flow']].values
    sse = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        sse[k] = kmeans.inertia_
    
    return sse


def plot_detector_map(detectors_df, anomaly_detid=None, zoom_start=11):
    """
    Create interactive Folium map of detectors.
    
    Parameters:
    -----------
    detectors_df : pd.DataFrame
        Detector dataframe with lat/long columns
    anomaly_detid : str, optional
        Detector ID to highlight as anomaly (larger blue circle)
    zoom_start : int
        Initial zoom level (default: 11)
    
    Returns:
    --------
    folium.Map
        Interactive Folium map object
    
    Example:
    --------
    >>> map_obj = plot_detector_map(detectors, 'CNTR_N03/164a1')
    >>> map_obj.save('detector_map.html')
    """
    # Center on London
    center_lat, center_long = 51.550929, -0.021497
    
    map_obj = folium.Map(
        location=[center_lat, center_long],
        zoom_start=zoom_start
    )
    
    # Plot all detectors as red circles
    for idx, row in detectors_df.iterrows():
        folium.Circle(
            radius=0.5,
            location=[row['lat'], row['long']],
            popup=row['detid'],
            color='crimson',
            fill=True,
            fillOpacity=0.7
        ).add_to(map_obj)
    
    # Highlight anomaly detector in blue if provided
    if anomaly_detid:
        anomaly = detectors_df[detectors_df['detid'] == anomaly_detid]
        if len(anomaly) > 0:
            anomaly_row = anomaly.iloc[0]
            folium.Circle(
                radius=3,
                location=[anomaly_row['lat'], anomaly_row['long']],
                popup=f"Anomaly: {anomaly_detid}",
                color='blue',
                fill=True,
                fillOpacity=0.8
            ).add_to(map_obj)
    
    return map_obj


def plot_clusters(data_df, labels, centroids, title='K-Means Clustering Results'):
    """
    Visualize K-Means clustering results.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame with 'mean_occ' and 'mean_flow' columns
    labels : np.ndarray
        Cluster labels from K-Means
    centroids : np.ndarray
        Cluster centroids
    title : str
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    
    Example:
    --------
    >>> fig = plot_clusters(stats, labels, centroids)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        data_df['mean_occ'],
        data_df['mean_flow'],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black'
    )
    
    # Plot centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidth=2,
        label='Centroids'
    )
    
    ax.set_xlabel('Mean Occupancy (%)', fontsize=12)
    ax.set_ylabel('Mean Flow (vehicles)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    return fig


if __name__ == '__main__':
    """
    Example usage of traffic_utils functions
    """
    # Load data
    detectors = load_and_filter_detectors('data/detectors_public.csv', 'london')
    print(f"Loaded {len(detectors)} London detectors")
    
    # Find nearest neighbors
    neighbors = find_nearest_detectors(detectors, 'CNTR_N03/164a1', 5)
    print(f"5 nearest detectors to CNTR_N03/164a1: {neighbors}")
    
    # Plot map
    traffic_map = plot_detector_map(detectors, 'CNTR_N03/164a1')
    print("Map created successfully")
