# Dataset Files

## About the Data

This folder contains the **UTD19 Multi-City Traffic Detector Dataset** from ETH Zurich.

**Note**: Data files are NOT included in this repository because they are too large (2.3 GB+). You must download them separately.

## Download Instructions

1. Visit: **[https://utd19.ethz.ch/](https://utd19.ethz.ch/)**
2. Download the following files and place them in this `data/` folder:

### Required Files:
- **`detectors_public.csv`** - Detector metadata (locations, IDs, city codes)
  - Contains: ~2,000 detectors across multiple European cities
  - Size: ~2.5 MB

- **`data_london.csv`** - Raw traffic measurements for London detectors
  - Contains: Millions of records with flow, occupancy, speed
  - Size: ~2.3 GB (large file - takes 1-2 minutes to load)

### Optional Files:
- **`mean_detectors.csv`** - Pre-computed statistics for faster analysis (~42 KB)
- **`traffic.csv`** - Additional traffic data (~1.7 MB)
- **`links.csv`** - Road network topology (~7.2 MB)

## Folder Structure After Setup

```
data/
├── README.md (this file)
├── detectors_public.csv     (required)
├── data_london.csv          (required)
├── mean_detectors.csv       (optional)
├── traffic.csv              (optional)
└── links.csv                (optional)
```

## Why Not in Repository?

GitHub has a **100 MB file size limit** per file. Since `data_london.csv` is 2.3 GB, it cannot be stored in Git. This is standard practice - data files are typically downloaded separately from the source.

## Relative Paths

All paths in the notebook use **relative paths** (e.g., `data/data_london.csv`), so the notebook will work correctly once you place the files in this folder.

## Questions?

- For data documentation: Visit [https://utd19.ethz.ch/](https://utd19.ethz.ch/)
- For project issues: See the main [README.md](../README.md)
