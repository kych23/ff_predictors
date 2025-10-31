# Fantasy Football Predictors - Frontend

Next.js frontend for the Fantasy Football Predictors application with a beautiful, modern UI.

## ✨ Features

- 🎨 **Modern Design**: Gradient-based UI with smooth animations and glass morphism effects
- 📈 **Predictions Leaderboard** - View predicted fantasy points for any season/week
- 🔍 **Smart Player Comparison** - Compare two players head-to-head with visual indicators
- 🏷️ **Color-Coded Positions** - Visual position badges (QB, RB, WR, TE)
- 🏆 **Top Player Highlighting** - Medals and special styling for top 3 players
- 📱 **Responsive Design** - Mobile-first, works on all devices
- 🌙 **Dark Mode** - Automatic theme support
- ⚡ **Fast Performance** - Built with Next.js 16 and React 19
- 🎯 **Animated Interactions** - Smooth transitions and hover effects

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000

### Installation

```bash
cd frontend
npm install
```

### Environment Variables

Create a `.env.local` file:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## 🎨 Design Features

### Visual Elements

- **Gradient Text**: Eye-catching gradient headings
- **Glass Morphism**: Translucent cards with backdrop blur
- **Position Badges**: Color-coded badges for QB (blue), RB (green), WR (purple), TE (orange)
- **Top 3 Medals**: 🥇🥈🥉 for top performers
- **VS Divider**: Pulsing comparison indicator
- **Winner Card**: Prominent green winner announcement

### Animations

- Fade-in-up on page load
- Smooth hover transitions
- Pulsing glow effects
- Shimmer loading states
- Staggered table row animations

## 📄 Pages

- **`/`** - Predictions leaderboard with filtering by season, week, and position
- **`/compare`** - Compare two players with head-to-head predictions and winner

## 🛠️ Tech Stack

- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS 4** - Utility-first styling
- **Custom CSS** - Advanced animations and effects
- **FastAPI Backend** - REST API

## 🎯 User Experience

The UI is designed for fantasy football managers who need to:

1. Quickly scan top projections
2. Compare players to make start/sit decisions
3. See confidence intervals for risk assessment
4. Filter by position for lineup construction
5. Make decisions on mobile devices

## 🔧 Customization

Edit `app/globals.css` for:

- Color schemes and gradients
- Animation timings
- Position badge colors
- Glass effect intensity
