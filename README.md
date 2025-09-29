# Amazon Recommender using Two Tower Recall

## Project Overview
This project implements a two-tower neural network architecture for Amazon product recommendation using recall-based retrieval. The system consists of a user tower and an item tower that learn embeddings to enable efficient candidate retrieval.


## Project Structure

```
Amazon_Recommender_using_Two_Tower_Recall/
├── Machine Learning Core
│   ├── src/                   # Core algorithm code
│   │   ├── models/            # Two-tower model implementation
│   │   ├── data/              # Data loader
│   │   ├── training/          # Training module
│   │   └── utils/             # Utility functions
│   ├── data/                  # Data storage
│   │   ├── raw/              # Raw data
│   │   └── processed/        # Preprocessed data  
│   ├── notebooks/            # Data exploration
│   │   ├── exploration/      # Data exploration
│   │   └── analysis/         # Data analysis
│   ├── config/               # Configuration files
│   ├── scripts/              # Execution scripts
│   │   ├── data_processing/  # Data preprocessing
│   │   └── training/         # Model training
│   ├── results/              # Output results
│   │   ├── models/           # Trained models
│   │   ├── metrics/          # Evaluation metrics
│   │   └── plots/            # Visualization plots
│   └── tests/                # Test files
│       ├── unit/             # Unit tests
│       └── integration/      # Integration tests
│
├── Frontend Application
│   └── frontend/             # React frontend
│       ├── src/              # Source code
│       │   ├── components/   # React components
│       │   ├── pages/        # Page components
│       │   ├── services/     # API services
│       │   ├── styles/       # Style files
│       │   └── utils/        # Utility functions
│       └── public/           # Static assets
│
├── Backend API
│   └── backend/              # FastAPI backend
│       └── app/              # Source code
│           ├── models/       # Data models
│           ├── routes/       # API routes
│           ├── services/     # Business services
│           └── utils/        # Utility functions
│
├── Documentation
│   └── docs/                 # Project documentation
│
└── Project Files
    └── README.md             # Project description
    

```
