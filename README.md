# Urban Traffic Analysis with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Comprehensive analysis of urban traffic flow using the **UTD19 Multi-City Traffic Detector Dataset** from ETH Zurich. This project demonstrates an end-to-end machine learning pipeline including data engineering, geospatial visualization, clustering analysis, and neural network-based predictive modeling for intelligent transportation systems.

## Project Overview

This project analyzes the largest open-source multi-city traffic detector loop dataset from ETH Zurich, with a primary focus on **London's traffic network**. The analysis demonstrates:

- **Data Engineering**: Processing and filtering large-scale traffic datasets (millions of records)
- **Geospatial Visualization**: Interactive maps showing detector networks and identifying faulty sensors
- **Statistical Analysis**: Computing traffic patterns, occupancy rates, and flow correlations
- **Unsupervised Learning**: K-Means clustering to identify traffic behavior patterns
- **Deep Learning**: Custom PyTorch neural network for predicting traffic flow from nearby detectors
- **Anomaly Detection**: Identifying and handling faulty sensors using spatial analysis

## Dataset

**UTD19**: The largest freely available multi-city traffic detector loop dataset from ETH Zurich

- **Source**: [https://utd19.ethz.ch/](https://utd19.ethz.ch/)
- **Size**: Multi-gigabyte dataset covering multiple European cities
- **Cities**: London (primary focus), Amsterdam, Basel, and others
- **Time Period**: Multi-year historical traffic data
- **Sensors**: Inductive loop detectors measuring:
  - **Flow**: Number of vehicles passing per time interval
  - **Occupancy**: Percentage of time detector is occupied by vehicles
  - **Speed**: Vehicle velocity measurements
  - **Time Resolution**: Aggregated intervals (typically 5-15 minutes)

### Data Files

All data files are located in the `data/` folder:

#### `detectors_public.csv`
Metadata for all traffic detectors across multiple cities.
- **Columns**:
  - `detid`: Unique detector identifier (e.g., "CNTR_N03/164a1")
  - `citycode`: City code (e.g., "london", "basel")
  - `lat`: Latitude coordinate
  - `long`: Longitude coordinate
  - `road_type`: Type of road (motorway, urban, etc.)
- **Size**: ~2,000+ detectors across all cities
- **London Subset**: ~1,000+ detectors

#### `data_london.csv`
Raw traffic measurements for London detectors (primary analysis dataset).
- **Columns**:
  - `detid`: Detector identifier (links to detectors_public.csv)
  - `day`: Date of measurement
  - `interval`: Time interval within the day
  - `flow`: Number of vehicles counted
  - `occ`: Occupancy percentage (0-100%)
  - `speed`: Average speed (if available)
- **Size**: Large file (~100MB+, millions of records)
- **Note**: Loading takes 1-2 minutes

#### `mean_detectors.csv`
Pre-computed aggregated statistics for 100 London detectors (for faster analysis).
- **Columns**:
  - `mean_occ`: Mean occupancy across all time periods
  - `mean_flow`: Mean flow across all time periods
- **Purpose**: Used for clustering analysis and visualization
- **Size**: 100 rows (one per detector)

#### `traffic.csv`
Processed traffic statistics (additional analysis file).
- Contains aggregated traffic metrics
- Used for supplementary analysis

#### `links.csv`
Road network topology data.
- **Columns**: Connections between road segments
- **Purpose**: Network analysis and route planning
- **Usage**: Optional for advanced network-based analysis

### Downloading the Dataset

1. Visit [https://utd19.ethz.ch/](https://utd19.ethz.ch/)
2. Download the required files:
   - `detectors_public.csv` (required)
   - London traffic data (required)
   - Additional city data (optional)
3. Place files in the `data/` folder
4. The notebook will automatically load files using relative paths

## Key Features & Methodology

### 1. Data Loading & Preprocessing
- Load detector metadata from `detectors_public.csv`
- Filter for London-specific detectors (citycode == "london")
- Load traffic measurements from `data_london.csv`
- Handle large datasets efficiently with pandas

### 2. Geospatial Visualization
- **Interactive Maps**: Create Folium maps centered on London
- **Detector Plotting**: Visualize all detector locations with coordinates
- **Anomaly Highlighting**: Identify faulty detector (CNTR_N03/164a1) and highlight in blue
- **Interactive Elements**: Popup information for each detector

### 3. Faulty Sensor Analysis
- Identify problematic detector (CNTR_N03/164a1)
- Calculate Euclidean distances to all other detectors
- Find 5 nearest neighboring detectors for imputation
- Spatial correlation analysis

### 4. Statistical Analysis
- Compute mean occupancy and flow for 100 detectors
- Analyze relationships between traffic metrics
- Create scatter plots showing occupancy vs. flow patterns
- Identify traffic behavior clusters

### 5. Machine Learning: K-Means Clustering
- Apply K-Means algorithm (7 clusters) to traffic patterns
- Visualize clusters with color-coded scatter plots
- **Elbow Method**: Determine optimal number of clusters
- Interpret different traffic behavior groups:
  - High flow, low occupancy (free-flowing traffic)
  - High occupancy, low flow (congestion)
  - Other intermediate patterns

### 6. Deep Learning: Neural Network for Traffic Prediction
- **Problem**: Predict occupancy of faulty detector using nearby detectors
- **Architecture**: 
  - Input: 5 features (occupancy from 5 nearest detectors)
  - Hidden layers: 32 → 16 → 16 neurons
  - Output: 1 value (predicted occupancy)
  - Activation: ReLU
- **Training**: 100 epochs using Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation**: Track training loss over epochs

### 7. Data Visualization
- **Interactive Plots**: Plotly scatter plots for exploration
- **Static Plots**: Matplotlib for publication-quality figures
- **Maps**: Folium for geospatial visualization
- **Network Architecture**: Custom visualization of neural network structure

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- 4GB+ RAM recommended for data processing

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/UTD19-Traffic-Analysis.git
cd UTD19-Traffic-Analysis
```

#### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Dataset
Place the UTD19 dataset files in the `data/` directory:

**Required files:**
- `detectors_public.csv` (detector metadata)
- `data_london.csv` (London traffic measurements)

**Optional pre-computed files:**
- `mean_detectors.csv` (aggregated statistics)
- `traffic.csv`, `links.csv` (additional data)

Download from: [https://utd19.ethz.ch/](https://utd19.ethz.ch/)

#### 5. Launch Jupyter Notebook
```bash
jupyter notebook traffic_analysis.ipynb
```

### Using the Utility Module

**Note:** The `traffic_utils.py` module is **OPTIONAL**. The entire analysis can be completed using only `traffic_analysis.ipynb`. 

The utility module exists to demonstrate:
- **Software engineering practices**: Modular, reusable code structure
- **Professional Python development**: Type hints, docstrings, error handling
- **Code reusability**: Functions can be imported and used in other projects
- **Testability**: Unit tests can be written for isolated functions

If you prefer to work entirely within the notebook, simply run `traffic_analysis.ipynb` - no additional dependencies or modules needed.

If you want to use the utility functions:

```python
from traffic_utils import load_and_filter_detectors, apply_kmeans_clustering

# Load London detectors
detectors = load_and_filter_detectors('data/detectors_public.csv', 'london')

# Apply K-Means clustering
kmeans, labels, centroids = apply_kmeans_clustering(stats_data, n_clusters=7)
```

See [traffic_utils.py](traffic_utils.py) for complete documentation of all functions.

## Usage

Execute the notebook cells sequentially from top to bottom:

### Analysis Pipeline

1. **Data Loading & Filtering**
   - Import detector metadata from CSV
   - Filter for London detectors only
   - Display sample data

2. **Geospatial Visualization** 
   - Create interactive map centered on London
   - Plot all detector locations as red circles
   - Identify and highlight faulty detector (CNTR_N03/164a1) in blue

3. **Spatial Analysis**
   - Calculate distances between detectors
   - Find 5 nearest detectors to the faulty sensor
   - Prepare for imputation/prediction

4. **Traffic Data Analysis**
   - Load large traffic measurement dataset
   - Compute mean occupancy and flow per detector
   - Generate statistical summaries

5. **Data Visualization**
   - Create scatter plots (occupancy vs. flow)
   - Interactive Plotly visualizations
   - Identify traffic patterns visually

6. **Machine Learning: Clustering**
   - Apply K-Means clustering (7 clusters)
   - Visualize cluster assignments
   - Use elbow method to validate cluster count
   - Interpret traffic behavior groups

7. **Deep Learning: Predictive Modeling**
   - Prepare training dataset (5 inputs → 1 output)
   - Design neural network architecture
   - Train model for 100 epochs
   - Monitor training loss
   - Visualize learning progress

## Results & Outputs

The analysis generates:

### Visualizations
- **Interactive Maps**: Folium maps showing detector network and anomalies
- **Scatter Plots**: Plotly visualizations of occupancy vs. flow relationships
- **Cluster Plots**: Matplotlib visualization of K-Means clusters
- **Elbow Plot**: Optimal cluster determination
- **Neural Network Diagram**: Architecture visualization
- **Training Curves**: Loss over epochs

### Insights
- **~1,000 London detectors** analyzed and mapped
- **Traffic patterns identified** through clustering (congestion, free-flow, intermediate)
- **Faulty sensor detected** and nearest neighbors found for data imputation
- **Predictive model trained** to estimate missing detector values
- **Statistical relationships** between occupancy and flow documented

### Generated Files
- `neural_network_architecture.png`: Visual representation of the neural network
- Saved model checkpoints (if implemented)
- Processed data frames for further analysis

## Project Structure

```
UTD19-Traffic-Analysis/
├── README.md                          # Project documentation (this file)
├── traffic_analysis.ipynb             # Main analysis notebook (48 cells)
├── traffic_utils.py                   # Reusable utility functions
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
├── .gitignore                         # Git ignore rules
└── data/                              # Dataset folder (download from ETH Zurich)
    ├── README.md                      # Data setup instructions
    └── .gitkeep                       # Keeps folder structure in Git
```

### File Descriptions:
- **`traffic_analysis.ipynb`**: Main analysis notebook with visualizations and results
- **`traffic_utils.py`**: Reusable Python module with utility functions:
  - `load_and_filter_detectors()` - Load detector metadata
  - `find_nearest_detectors()` - Find nearest neighbors
  - `compute_detector_statistics()` - Calculate traffic statistics
  - `apply_kmeans_clustering()` - K-Means analysis
  - `plot_detector_map()` - Create interactive maps
  - `plot_clusters()` - Visualize clustering results
- **`data/` folder**: Empty in repository. Users download files from [https://utd19.ethz.ch/](https://utd19.ethz.ch/) and place them here
- **Relative paths**: All paths use `data/filename.csv` format (portable across systems)

## Technical Highlights

### Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Folium**: Interactive geospatial visualizations
- **Plotly**: Interactive data visualizations
- **Matplotlib**: Static plotting and diagrams
- **Scikit-learn**: K-Means clustering and ML utilities
- **PyTorch**: Deep learning framework for neural networks
- **Jupyter**: Interactive development environment

### Key Algorithms
1. **Euclidean Distance**: Spatial proximity calculation
2. **K-Means Clustering**: Unsupervised pattern recognition
3. **Elbow Method**: Cluster optimization
4. **Neural Network**: Multi-layer perceptron with ReLU activation
5. **Gradient Descent**: Adam optimizer for training

### Performance Considerations
- **Large Dataset**: `data_london.csv` loading takes 1-2 minutes
- **Memory Usage**: ~4GB RAM recommended for full analysis
- **Training Time**: Neural network trains in ~5-10 minutes on CPU
- **Portable**: All file paths are relative (works on any system)

## Future Enhancements

Potential areas for extension:
- **Real-time Prediction**: Stream processing for live traffic data
- **Multi-city Analysis**: Extend to Amsterdam, Basel, and other cities
- **Time Series Forecasting**: LSTM/GRU networks for temporal predictions
- **Web Dashboard**: Interactive Streamlit/Dash application
- **Advanced Models**: Attention mechanisms, Graph Neural Networks
- **Mobile App**: Real-time traffic monitoring application

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- Additional visualizations
- Performance optimizations
- Documentation improvements
- Bug fixes and testing
- New analysis techniques

## Technologies & Tools

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Folium (maps), Plotly (interactive), Matplotlib (static) |
| **Machine Learning** | Scikit-learn (K-Means), PyTorch (Neural Networks) |
| **Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

## Skills Demonstrated

This project showcases proficiency in:

### Data Science
- **Data Engineering**: Loading, cleaning, and processing large datasets (100MB+)
- **Statistical Analysis**: Computing means, correlations, and distributions
- **Feature Engineering**: Creating meaningful features from raw sensor data
- **Data Visualization**: Creating publication-quality static and interactive plots

### Machine Learning
- **Unsupervised Learning**: K-Means clustering for pattern recognition
- **Model Selection**: Elbow method for hyperparameter tuning
- **Deep Learning**: Custom PyTorch neural network architecture
- **Training & Optimization**: Gradient descent, loss tracking, convergence analysis

### Geospatial Analysis
- **Coordinate Systems**: Working with latitude/longitude data
- **Distance Calculations**: Euclidean distance for spatial proximity
- **Interactive Mapping**: Folium for web-based map visualizations
- **Spatial Queries**: Finding nearest neighbors and spatial patterns

### Software Engineering
- **Clean Code**: Well-documented, modular notebook structure
- **Version Control**: Git/GitHub best practices
- **Reproducibility**: Relative paths, requirements.txt, clear documentation
- **Project Organization**: Logical file structure and naming conventions

## Acknowledgments

- **ETH Zurich** - For providing the UTD19 dataset ([https://utd19.ethz.ch/](https://utd19.ethz.ch/))
- **Open Source Community** - For the excellent libraries: Pandas, PyTorch, Folium, Scikit-learn, Plotly, Matplotlib

- **Acknowledgments:** This project was completed as part of the "Data Science and AI in Intelligent and Sustainable Mobility Systems" course at Technische Hochschule Ingolstadt under the supervision of Prof. Dr. Stefanie Schmidtner.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Star this repository if you found it helpful!  
Fork it to create your own traffic analysis project!  
Issues and PRs are welcome!

