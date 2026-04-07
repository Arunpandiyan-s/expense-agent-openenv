# Adaptive Financial Decision Environment

A production-grade OpenEnv environment for evaluating AI agents on complex financial decision-making tasks.

## Overview

This environment simulates real-world expense analysis where an AI agent must:
- **Detect hidden financial leaks** in user expenses
- **Handle uncertainty** and incomplete information
- **Navigate trade-offs** between competing priorities
- **Discover hidden patterns** in spending behavior

Unlike simple rule-based systems, this environment requires genuine reasoning.

## Why It Matters

Evaluating AI reasoning on financial tasks is challenging because:
1. Real expenses have **ambiguous signals** (is a low-usage gym membership wasteful or a health investment?)
2. Small daily costs create **hidden patterns** ($6 coffee × 30 days = $180/month leak)
3. Decisions involve **trade-offs** (cutting food delivery saves money but costs time)
4. Information is often **incomplete** (usage data may be missing)

This environment captures these complexities in a reproducible, graded benchmark.

## How It Works

```
┌─────────────────────────────────────────────────┐
│                 ENVIRONMENT                      │
│  ┌───────────┐    ┌───────────┐    ┌──────────┐ │
│  │  Expenses │ →  │   Agent   │ →  │  Reward  │ │
│  │  + Profile│    │  Decision │    │  Signal  │ │
│  └───────────┘    └───────────┘    └──────────┘ │
│        ↑                                  │      │
│        └──────────────────────────────────┘      │
│                   feedback loop                  │
└─────────────────────────────────────────────────┘
```

1. **reset()** → Returns initial observation with expenses, budget, user profile
2. **step(action)** → Executes action, returns (observation, reward, done, info)
3. **Grader** → Evaluates final performance (0.0 to 1.0)

## Action Space

| Action | Description |
|--------|-------------|
| `analyze_pattern` | Analyze spending patterns (start here) |
| `flag_expense` | Flag an expense as a financial leak |
| `suggest_optimization` | Suggest a cost-saving optimization |
| `ask_clarification` | Request more info about an expense |
| `compare_expenses` | Compare multiple expenses |
| `investigate_pattern` | Investigate hidden patterns |
| `stop_analysis` | End analysis (when confident) |

## Observation Space

The agent observes:
- **Expenses**: ID, description, amount, category, vendor, usage score, flags
- **User Profile**: Income level, lifestyle, financial goals, debt level
- **State**: Step count, leaks found, suggestions made, efficiency score
- **Feedback**: Last action result, insights discovered

Ground truth (is_leak, leak_type) is hidden from the agent.

## Task Design

### Easy
Clear signals: unused Netflix (10% usage), duplicate music services, forgotten gym membership.
- 3 expected leaks, 10 max steps

### Medium
Ambiguous cases: $189 gym with 25% usage (overpriced or health priority?), daily coffee habit ($195/month hidden pattern), food delivery lifestyle.
- 6 expected leaks, 12 max steps

### Hard
Complex scenarios: cloud storage sprawl, hidden bank fees, goal-misaligned spending, conflicting signals requiring clarification.
- 10 expected leaks, 15 max steps

## Reward Design

| Event | Reward |
|-------|--------|
| Correct leak detection | +0.5 |
| Correct leak type | +0.1 |
| Pattern discovery | +0.2 |
| Useful suggestion | +0.3 |
| Incorrect flag | -0.3 |
| Repeated action | -0.2 |
| Useless action | -0.1 |
| Early stop (premature) | -0.2 |

## Grading

Final score (0.0 to 1.0) based on:
- **Accuracy** (30-50%): Precision, recall, F1 for leak detection
- **Efficiency** (15-20%): Steps used, redundant actions avoided
- **Reasoning** (20-30%): Quality of explanations and suggestions
- **Discovery** (15-25%): Hidden patterns found

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

1. Create `.env` file:
```
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
HF_TOKEN=your_token_here
```

2. Run inference:
```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

## Docker

```bash
docker build -t expense-env .
docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... expense-env --task easy
```

## Hugging Face Deployment

1. Create a new Space (Docker SDK)
2. Add secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
3. Push this repository

## Sample Output

```
[START] task=easy env=adaptive-financial-decision model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=analyze_pattern reward=0.15 done=false error=null
[STEP] step=2 action=flag_expense(e001) reward=0.60 done=false error=null
[STEP] step=3 action=flag_expense(e003) reward=0.50 done=false error=null
[STEP] step=4 action=flag_expense(e004) reward=0.50 done=false error=null
[STEP] step=5 action=suggest_optimization(Consolidate...) reward=0.30 done=false error=null
[STEP] step=6 action=stop_analysis reward=0.10 done=true error=null
[END] success=true steps=6 score=0.85 rewards=[0.15,0.60,0.50,0.50,0.30,0.10]
```

## Project Structure

```
expense-tracker/
├── env/
│   ├── __init__.py
│   ├── environment.py    # Core environment logic
│   ├── models.py         # Pydantic data models
│   ├── tasks.py          # Task definitions (easy/medium/hard)
│   └── grader.py         # Multi-dimensional grading
├── inference.py          # Production inference script
├── openenv.yaml          # Environment specification
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## License

MIT
