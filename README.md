# Fantasy Football Predictors

Machine learning project all about PPR fantasy football predictions. Predict fantasy points using XGBoost models trained on historical NFL data.

## 🏗️ Stack

**Backend:**

- Python for ETL/ML
- FastAPI for REST API
- PostgreSQL for storage
- XGBoost for predictions

**Frontend:**

- Next.js 16 (React 19)
- TypeScript
- Tailwind CSS

## 🚀 Quick Start

### Backend Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2. Start PostgreSQL (Docker or local service)

3. Initialize database:

```bash
python -c "from src.db.init_db import init_db; init_db(); print('DB ready')"
```

4. Load NFL data:

```bash
python scripts/seed_db.py --start 2012 --end 2025
```

5. Train models:

```bash
python scripts/test_train_pipeline.py
# With hyperparameter tuning:
python scripts/test_train_pipeline.py --tune
# Save predictions to DB:
python scripts/test_train_pipeline.py --save
```

6. Start API server:

```bash
uvicorn src.api.main:app --reload
```

### Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Configure environment (create `frontend/.env.local`):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 📊 Features

- 🎨 **Modern UI**: Beautiful gradient-based design with smooth animations
- 📈 **Predictions Leaderboard**: View top projected players with medals for top 3
- 🆚 **Player Comparison**: Visual head-to-head comparison with clear winner
- 🔍 **Smart Search**: Instant player search with autocomplete
- 🏷️ **Position Filtering**: Color-coded position badges (QB, RB, WR, TE)
- 📊 **Confidence Intervals**: See 95% prediction ranges
- 📱 **Responsive Design**: Works perfectly on mobile, tablet, and desktop
- 🌙 **Dark Mode**: Automatic dark mode support
- 📈 **Historical Analysis**: Query any season from 2012-2024

## 📚 API Endpoints

- `GET /predictions?season={year}&week={week}&position={pos}` - Get predictions
- `GET /players?position={pos}` - List players
- `GET /weeks?season={year}` - Get available weeks
- `POST /predict` - Get prediction for specific player

## 🎯 Model Performance

Current XGBoost models achieve:

- **QB**: MAE ~6.13, Within-3.0 Acc ~31%
- **Skill Positions (RB/WR/TE)**: MAE ~4.56, Within-3.0 Acc ~45%

## 📝 Training Pipeline

The training pipeline uses:

- Expanding time-series cross-validation (no data leakage)
- Position-specific models (QB vs SKILL)
- Randomized hyperparameter search
- Out-of-fold predictions for evaluation

## 🏆 Attribution

- NFL data via [nflverse](https://www.nflverse.com/) (`nflreadpy`)
