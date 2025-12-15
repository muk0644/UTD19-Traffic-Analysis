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

