# ML Recommendation System

Hybrid product recommendation system using collaborative filtering on real user interaction data.

## Features
- Event tracking (views, carts, purchases, searches)
- SQLite data store with indexes for performance
- Implicit library for ALS-based CF
- FastAPI ready for serving recommendations
- Modular structure for features/models/evaluation

## Quickstart

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize DB & Seed Data**
   ```bash
   python src/ingestion/event_tracker.py
   ```

3. **Development**
   - Notebooks: `jupyter notebook notebooks/`
   - API: Develop in `src/api/` (stub ready)
   - Models: Train in `src/models/`

## Architecture
```
data/raw/ → ingestion → data/processed/ → features → models → evaluation
                                      ↓
                                   api/
```
- **Ingestion**: event_tracker.py (live logging)
- **Features**: User-item matrix from interactions
- **Models**: Implicit ALS + content features
- **API**: FastAPI /recommend/{user_id}?n=10

## Next Steps
- Implement model training (`src/models/train.py`)
- Feature engineering (`src/features/`)
- API endpoints (`src/api/main.py`)
- Evaluation metrics (precision@K, NDCG)

See TODO.md for current plan progress.
