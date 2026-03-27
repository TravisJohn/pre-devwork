# House Price Prediction Project

A machine learning project to predict house prices using historical data.

## Project Structure
```
house_price_prediction/
├── venv/           # Project virtual environment
├── notebooks/      # For Jupyter exploration
├── src/           # For final production code
├── data/          # For your clean_house_data.csv
├── models/        # For saved pickle files
└── requirements-dev.txt
```

## Environment Setup

### Local Development
1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Deployment
- Deployment environment will use a minimal set of dependencies
- Model will be deployed using pickle files
- Separate requirements.txt will be created for production

## Development Workflow
1. Data exploration and model development in Jupyter notebooks
2. Transfer final code to `src/` directory
3. Train model locally and save as pickle file
4. Deploy pickled model to cloud environment

## Project Goals
- Develop accurate house price prediction model
- Create maintainable, production-ready code
- Implement efficient deployment pipeline
- Provide clear documentation for future maintenance

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/TravisJohn/house-price-prediction.git
cd house_price_prediction
```

2. Set up development environment as described above
3. Begin exploration in notebooks directory
4. Follow standard Git workflow for contributions

## Next Steps
- [ ] Install required development packages
- [ ] Begin model exploration using Jupyter notebooks
- [ ] Implement initial model training pipeline
- [ ] Create deployment strategy