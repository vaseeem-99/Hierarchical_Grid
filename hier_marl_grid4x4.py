"""
hier_marl_grid4x4.py — H-PPO and H-DQN for 4×4 Grid Traffic Control
=====================================================================
Algorithms : H-PPO (Hierarchical PPO) · H-DQN (Hierarchical DQN)
Features   : Dynamic traffic demand each episode
             Weights & Biases experiment tracking
             All plots identical to flat_marl results

Usage:
  python hier_marl_grid4x4.py --algo hppo  --episodes 300
  python hier_marl_grid4x4.py --algo hdqn  --episodes 300
  python hier_marl_grid4x4.py --algo all   --episodes 300
  python hier_marl_grid4x4.py --algo hppo  --no-wandb
  python hier_marl_grid4x4.py --compare    # compare H-PPO vs H-DQN
"""

import os, sys, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# 0. SUMO
# ══════════════════════════════════════════════════════════════════════════════

if "SUMO_HOME" not in os.environ:
    for _p in ["/usr/share/sumo","/opt/homebrew/share/sumo",
               "C:/Program Files (x86)/Eclipse/Sumo","C:/Program Files/Eclipse/Sumo"]:
        if os.path.exists(_p): os.environ["SUMO_HOME"]=_p; break

sys.path += [os.path.join(os.environ.get("SUMO_HOME",""),"tools")]
import traci

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(BASE_DIR,"grid4x4.net.xml")

TLS_IDS=["A0","A1","A2","A3","B0","B1","B2","B3",
         "C0","C1","C2","C3","D0","D1","D2","D3"]
NUM_AGENTS=16
REGIONS={"R_A":["A0","A1","A2","A3"],"R_B":["B0","B1","B2","B3"],
         "R_C":["C0","C1","C2","C3"],"R_D":["D0","D1","D2","D3"]}
REGION_IDS=list(REGIONS.keys()); NUM_REGIONS=4
TLS_TO_REGION={tls:reg for reg,members in REGIONS.items() for tls in members}

GREEN_PHASES=[0,2]; YELLOW_DUR=3; DELTA_T=5; SIM_STEPS=3600
MAX_QUEUE=30; MAX_GREEN=60; LANES_PER_EDGE=3; NUM_PHASES=2
DIRECTIVE_DIM=4
LOCAL_OBS_DIM=27
REGIONAL_OBS_DIM=LOCAL_OBS_DIM*4   # 108
GLOBAL_OBS_DIM=LOCAL_OBS_DIM*NUM_AGENTS  # 432
LOCAL_INPUT_DIM=LOCAL_OBS_DIM+DIRECTIVE_DIM   # 31
REGIONAL_INPUT_DIM=REGIONAL_OBS_DIM+DIRECTIVE_DIM  # 112
SCENARIOS=["original","light","medium","heavy","uneven"]

def _build_incoming_edges():
    edges={}; cols=["A","B","C","D"]
    for ci,col in enumerate(cols):
        for row in range(4):
            tls=f"{col}{row}"; inc=[]
            inc.append(f"left{row}A{row}"   if ci==0 else f"{cols[ci-1]}{row}{col}{row}")
            inc.append(f"right{row}D{row}"  if ci==3 else f"{cols[ci+1]}{row}{col}{row}")
            inc.append(f"bottom{ci}{col}0"  if row==0 else f"{col}{row-1}{col}{row}")
            inc.append(f"top{ci}{col}3"     if row==3 else f"{col}{row+1}{col}{row}")
            edges[tls]=inc
    return edges

INCOMING_EDGES=_build_incoming_edges()
AGENT_LANES={tls:[f"{e}_{l}" for e in INCOMING_EDGES[tls] for l in range(LANES_PER_EDGE)]
             for tls in TLS_IDS}

# ══════════════════════════════════════════════════════════════════════════════
# 2. DYNAMIC DEMAND
# ══════════════════════════════════════════════════════════════════════════════

ROUTE_MAP = {
    ("bottom0A0", "A0bottom0"): "bottom0A0 A0A1 A1B1 B1B0 B0A0 A0bottom0",
    ("bottom0A0", "A0left0"): "bottom0A0 A0left0",
    ("bottom0A0", "A1left1"): "bottom0A0 A0A1 A1left1",
    ("bottom0A0", "A2left2"): "bottom0A0 A0A1 A1A2 A2left2",
    ("bottom0A0", "A3left3"): "bottom0A0 A0A1 A1A2 A2A3 A3left3",
    ("bottom0A0", "A3top0"): "bottom0A0 A0A1 A1A2 A2A3 A3top0",
    ("bottom0A0", "B0bottom1"): "bottom0A0 A0B0 B0bottom1",
    ("bottom0A0", "B3top1"): "bottom0A0 A0B0 B0B1 B1B2 B2B3 B3top1",
    ("bottom0A0", "C0bottom2"): "bottom0A0 A0B0 B0C0 C0bottom2",
    ("bottom0A0", "C3top2"): "bottom0A0 A0A1 A1A2 A2A3 A3B3 B3C3 C3top2",
    ("bottom0A0", "D0bottom3"): "bottom0A0 A0B0 B0C0 C0D0 D0bottom3",
    ("bottom0A0", "D0right0"): "bottom0A0 A0B0 B0C0 C0D0 D0right0",
    ("bottom0A0", "D1right1"): "bottom0A0 A0A1 A1B1 B1C1 C1D1 D1right1",
    ("bottom0A0", "D2right2"): "bottom0A0 A0A1 A1A2 A2B2 B2C2 C2D2 D2right2",
    ("bottom0A0", "D3right3"): "bottom0A0 A0A1 A1A2 A2A3 A3B3 B3C3 C3D3 D3right3",
    ("bottom0A0", "D3top3"): "bottom0A0 A0A1 A1A2 A2A3 A3B3 B3C3 C3D3 D3top3",
    ("bottom1B0", "A0bottom0"): "bottom1B0 B0A0 A0bottom0",
    ("bottom1B0", "A0left0"): "bottom1B0 B0A0 A0left0",
    ("bottom1B0", "A1left1"): "bottom1B0 B0B1 B1A1 A1left1",
    ("bottom1B0", "A2left2"): "bottom1B0 B0B1 B1B2 B2A2 A2left2",
    ("bottom1B0", "A3left3"): "bottom1B0 B0B1 B1B2 B2B3 B3A3 A3left3",
    ("bottom1B0", "A3top0"): "bottom1B0 B0B1 B1A1 A1A2 A2A3 A3top0",
    ("bottom1B0", "B0bottom1"): "bottom1B0 B0B1 B1C1 C1C0 C0B0 B0bottom1",
    ("bottom1B0", "B3top1"): "bottom1B0 B0B1 B1B2 B2B3 B3top1",
    ("bottom1B0", "C0bottom2"): "bottom1B0 B0C0 C0bottom2",
    ("bottom1B0", "C3top2"): "bottom1B0 B0C0 C0C1 C1C2 C2C3 C3top2",
    ("bottom1B0", "D0bottom3"): "bottom1B0 B0C0 C0D0 D0bottom3",
    ("bottom1B0", "D0right0"): "bottom1B0 B0C0 C0D0 D0right0",
    ("bottom1B0", "D1right1"): "bottom1B0 B0B1 B1C1 C1D1 D1right1",
    ("bottom1B0", "D2right2"): "bottom1B0 B0B1 B1B2 B2C2 C2D2 D2right2",
    ("bottom1B0", "D3right3"): "bottom1B0 B0B1 B1B2 B2B3 B3C3 C3D3 D3right3",
    ("bottom1B0", "D3top3"): "bottom1B0 B0B1 B1B2 B2B3 B3C3 C3D3 D3top3",
    ("bottom2C0", "A0bottom0"): "bottom2C0 C0B0 B0A0 A0bottom0",
    ("bottom2C0", "A0left0"): "bottom2C0 C0B0 B0A0 A0left0",
    ("bottom2C0", "A1left1"): "bottom2C0 C0C1 C1B1 B1A1 A1left1",
    ("bottom2C0", "A2left2"): "bottom2C0 C0C1 C1C2 C2B2 B2A2 A2left2",
    ("bottom2C0", "A3left3"): "bottom2C0 C0C1 C1C2 C2C3 C3B3 B3A3 A3left3",
    ("bottom2C0", "A3top0"): "bottom2C0 C0C1 C1B1 B1A1 A1A2 A2A3 A3top0",
    ("bottom2C0", "B0bottom1"): "bottom2C0 C0B0 B0bottom1",
    ("bottom2C0", "B3top1"): "bottom2C0 C0C1 C1B1 B1B2 B2B3 B3top1",
    ("bottom2C0", "C0bottom2"): "bottom2C0 C0C1 C1D1 D1D0 D0C0 C0bottom2",
    ("bottom2C0", "C3top2"): "bottom2C0 C0C1 C1C2 C2C3 C3top2",
    ("bottom2C0", "D0bottom3"): "bottom2C0 C0D0 D0bottom3",
    ("bottom2C0", "D0right0"): "bottom2C0 C0D0 D0right0",
    ("bottom2C0", "D1right1"): "bottom2C0 C0C1 C1D1 D1right1",
    ("bottom2C0", "D2right2"): "bottom2C0 C0C1 C1C2 C2D2 D2right2",
    ("bottom2C0", "D3right3"): "bottom2C0 C0C1 C1C2 C2C3 C3D3 D3right3",
    ("bottom2C0", "D3top3"): "bottom2C0 C0D0 D0D1 D1D2 D2D3 D3top3",
    ("bottom3D0", "A0bottom0"): "bottom3D0 D0C0 C0B0 B0A0 A0bottom0",
    ("bottom3D0", "A0left0"): "bottom3D0 D0C0 C0B0 B0A0 A0left0",
    ("bottom3D0", "A1left1"): "bottom3D0 D0D1 D1C1 C1B1 B1A1 A1left1",
    ("bottom3D0", "A2left2"): "bottom3D0 D0D1 D1D2 D2C2 C2B2 B2A2 A2left2",
    ("bottom3D0", "A3left3"): "bottom3D0 D0D1 D1D2 D2D3 D3C3 C3B3 B3A3 A3left3",
    ("bottom3D0", "A3top0"): "bottom3D0 D0D1 D1C1 C1B1 B1A1 A1A2 A2A3 A3top0",
    ("bottom3D0", "B0bottom1"): "bottom3D0 D0C0 C0B0 B0bottom1",
    ("bottom3D0", "B3top1"): "bottom3D0 D0D1 D1C1 C1B1 B1B2 B2B3 B3top1",
    ("bottom3D0", "C0bottom2"): "bottom3D0 D0C0 C0bottom2",
    ("bottom3D0", "C3top2"): "bottom3D0 D0D1 D1C1 C1C2 C2C3 C3top2",
    ("bottom3D0", "D0bottom3"): "bottom3D0 D0C0 C0C1 C1D1 D1D0 D0bottom3",
    ("bottom3D0", "D0right0"): "bottom3D0 D0right0",
    ("bottom3D0", "D1right1"): "bottom3D0 D0D1 D1right1",
    ("bottom3D0", "D2right2"): "bottom3D0 D0D1 D1D2 D2right2",
    ("bottom3D0", "D3right3"): "bottom3D0 D0D1 D1D2 D2D3 D3right3",
    ("bottom3D0", "D3top3"): "bottom3D0 D0D1 D1D2 D2D3 D3top3",
    ("left0A0", "A0bottom0"): "left0A0 A0bottom0",
    ("left0A0", "A0left0"): "left0A0 A0A1 A1B1 B1B0 B0A0 A0left0",
    ("left0A0", "A1left1"): "left0A0 A0A1 A1left1",
    ("left0A0", "A2left2"): "left0A0 A0A1 A1A2 A2left2",
    ("left0A0", "A3left3"): "left0A0 A0A1 A1A2 A2A3 A3left3",
    ("left0A0", "A3top0"): "left0A0 A0A1 A1A2 A2A3 A3top0",
    ("left0A0", "B0bottom1"): "left0A0 A0B0 B0bottom1",
    ("left0A0", "B3top1"): "left0A0 A0B0 B0B1 B1B2 B2B3 B3top1",
    ("left0A0", "C0bottom2"): "left0A0 A0B0 B0C0 C0bottom2",
    ("left0A0", "C3top2"): "left0A0 A0B0 B0C0 C0C1 C1C2 C2C3 C3top2",
    ("left0A0", "D0bottom3"): "left0A0 A0B0 B0C0 C0D0 D0bottom3",
    ("left0A0", "D0right0"): "left0A0 A0B0 B0C0 C0D0 D0right0",
    ("left0A0", "D1right1"): "left0A0 A0B0 B0B1 B1C1 C1D1 D1right1",
    ("left0A0", "D2right2"): "left0A0 A0B0 B0B1 B1B2 B2C2 C2D2 D2right2",
    ("left0A0", "D3right3"): "left0A0 A0B0 B0B1 B1B2 B2B3 B3C3 C3D3 D3right3",
    ("left0A0", "D3top3"): "left0A0 A0B0 B0C0 C0D0 D0D1 D1D2 D2D3 D3top3",
    ("left1A1", "A0bottom0"): "left1A1 A1A0 A0bottom0",
    ("left1A1", "A0left0"): "left1A1 A1A0 A0left0",
    ("left1A1", "A1left1"): "left1A1 A1B1 B1B0 B0A0 A0A1 A1left1",
    ("left1A1", "A2left2"): "left1A1 A1A2 A2left2",
    ("left1A1", "A3left3"): "left1A1 A1A2 A2A3 A3left3",
    ("left1A1", "A3top0"): "left1A1 A1A2 A2A3 A3top0",
    ("left1A1", "B0bottom1"): "left1A1 A1B1 B1B0 B0bottom1",
    ("left1A1", "B3top1"): "left1A1 A1B1 B1B2 B2B3 B3top1",
    ("left1A1", "C0bottom2"): "left1A1 A1B1 B1C1 C1C0 C0bottom2",
    ("left1A1", "C3top2"): "left1A1 A1B1 B1C1 C1C2 C2C3 C3top2",
    ("left1A1", "D0bottom3"): "left1A1 A1B1 B1C1 C1D1 D1D0 D0bottom3",
    ("left1A1", "D0right0"): "left1A1 A1A0 A0B0 B0C0 C0D0 D0right0",
    ("left1A1", "D1right1"): "left1A1 A1B1 B1C1 C1D1 D1right1",
    ("left1A1", "D2right2"): "left1A1 A1B1 B1B2 B2C2 C2D2 D2right2",
    ("left1A1", "D3right3"): "left1A1 A1B1 B1B2 B2B3 B3C3 C3D3 D3right3",
    ("left1A1", "D3top3"): "left1A1 A1B1 B1C1 C1D1 D1D2 D2D3 D3top3",
    ("left2A2", "A0bottom0"): "left2A2 A2A1 A1A0 A0bottom0",
    ("left2A2", "A0left0"): "left2A2 A2A1 A1A0 A0left0",
    ("left2A2", "A1left1"): "left2A2 A2A1 A1left1",
    ("left2A2", "A2left2"): "left2A2 A2B2 B2B1 B1A1 A1A2 A2left2",
    ("left2A2", "A3left3"): "left2A2 A2A3 A3left3",
    ("left2A2", "A3top0"): "left2A2 A2A3 A3top0",
    ("left2A2", "B0bottom1"): "left2A2 A2B2 B2B1 B1B0 B0bottom1",
    ("left2A2", "B3top1"): "left2A2 A2B2 B2B3 B3top1",
    ("left2A2", "C0bottom2"): "left2A2 A2B2 B2C2 C2C1 C1C0 C0bottom2",
    ("left2A2", "C3top2"): "left2A2 A2B2 B2C2 C2C3 C3top2",
    ("left2A2", "D0bottom3"): "left2A2 A2B2 B2C2 C2D2 D2D1 D1D0 D0bottom3",
    ("left2A2", "D0right0"): "left2A2 A2B2 B2C2 C2D2 D2D1 D1D0 D0right0",
    ("left2A2", "D1right1"): "left2A2 A2A1 A1B1 B1C1 C1D1 D1right1",
    ("left2A2", "D2right2"): "left2A2 A2B2 B2C2 C2D2 D2right2",
    ("left2A2", "D3right3"): "left2A2 A2B2 B2B3 B3C3 C3D3 D3right3",
    ("left2A2", "D3top3"): "left2A2 A2B2 B2C2 C2D2 D2D3 D3top3",
    ("left3A3", "A0bottom0"): "left3A3 A3A2 A2A1 A1A0 A0bottom0",
    ("left3A3", "A0left0"): "left3A3 A3A2 A2A1 A1A0 A0left0",
    ("left3A3", "A1left1"): "left3A3 A3A2 A2A1 A1left1",
    ("left3A3", "A2left2"): "left3A3 A3A2 A2left2",
    ("left3A3", "A3left3"): "left3A3 A3B3 B3B2 B2A2 A2A3 A3left3",
    ("left3A3", "A3top0"): "left3A3 A3top0",
    ("left3A3", "B0bottom1"): "left3A3 A3B3 B3B2 B2B1 B1B0 B0bottom1",
    ("left3A3", "B3top1"): "left3A3 A3B3 B3top1",
    ("left3A3", "C0bottom2"): "left3A3 A3B3 B3C3 C3C2 C2C1 C1C0 C0bottom2",
    ("left3A3", "C3top2"): "left3A3 A3B3 B3C3 C3top2",
    ("left3A3", "D0bottom3"): "left3A3 A3B3 B3C3 C3D3 D3D2 D2D1 D1D0 D0bottom3",
    ("left3A3", "D0right0"): "left3A3 A3B3 B3C3 C3D3 D3D2 D2D1 D1D0 D0right0",
    ("left3A3", "D1right1"): "left3A3 A3B3 B3C3 C3D3 D3D2 D2D1 D1right1",
    ("left3A3", "D2right2"): "left3A3 A3A2 A2B2 B2C2 C2D2 D2right2",
    ("left3A3", "D3right3"): "left3A3 A3B3 B3C3 C3D3 D3right3",
    ("left3A3", "D3top3"): "left3A3 A3B3 B3C3 C3D3 D3top3",
    ("right0D0", "A0bottom0"): "right0D0 D0C0 C0B0 B0A0 A0bottom0",
    ("right0D0", "A0left0"): "right0D0 D0C0 C0B0 B0A0 A0left0",
    ("right0D0", "A1left1"): "right0D0 D0D1 D1C1 C1B1 B1A1 A1left1",
    ("right0D0", "A2left2"): "right0D0 D0C0 C0B0 B0A0 A0A1 A1A2 A2left2",
    ("right0D0", "A3left3"): "right0D0 D0C0 C0B0 B0A0 A0A1 A1A2 A2A3 A3left3",
    ("right0D0", "A3top0"): "right0D0 D0C0 C0B0 B0A0 A0A1 A1A2 A2A3 A3top0",
    ("right0D0", "B0bottom1"): "right0D0 D0C0 C0B0 B0bottom1",
    ("right0D0", "B3top1"): "right0D0 D0C0 C0B0 B0B1 B1B2 B2B3 B3top1",
    ("right0D0", "C0bottom2"): "right0D0 D0C0 C0bottom2",
    ("right0D0", "C3top2"): "right0D0 D0C0 C0C1 C1C2 C2C3 C3top2",
    ("right0D0", "D0bottom3"): "right0D0 D0bottom3",
    ("right0D0", "D0right0"): "right0D0 D0C0 C0C1 C1D1 D1D0 D0right0",
    ("right0D0", "D1right1"): "right0D0 D0D1 D1right1",
    ("right0D0", "D2right2"): "right0D0 D0D1 D1D2 D2right2",
    ("right0D0", "D3right3"): "right0D0 D0D1 D1D2 D2D3 D3right3",
    ("right0D0", "D3top3"): "right0D0 D0D1 D1D2 D2D3 D3top3",
    ("right1D1", "A0bottom0"): "right1D1 D1C1 C1B1 B1A1 A1A0 A0bottom0",
    ("right1D1", "A0left0"): "right1D1 D1C1 C1C0 C0B0 B0A0 A0left0",
    ("right1D1", "A1left1"): "right1D1 D1C1 C1B1 B1A1 A1left1",
    ("right1D1", "A2left2"): "right1D1 D1D2 D2C2 C2B2 B2A2 A2left2",
    ("right1D1", "A3left3"): "right1D1 D1C1 C1B1 B1A1 A1A2 A2A3 A3left3",
    ("right1D1", "A3top0"): "right1D1 D1C1 C1B1 B1A1 A1A2 A2A3 A3top0",
    ("right1D1", "B0bottom1"): "right1D1 D1C1 C1B1 B1B0 B0bottom1",
    ("right1D1", "B3top1"): "right1D1 D1C1 C1B1 B1B2 B2B3 B3top1",
    ("right1D1", "C0bottom2"): "right1D1 D1C1 C1C0 C0bottom2",
    ("right1D1", "C3top2"): "right1D1 D1C1 C1C2 C2C3 C3top2",
    ("right1D1", "D0bottom3"): "right1D1 D1D0 D0bottom3",
    ("right1D1", "D0right0"): "right1D1 D1D0 D0right0",
    ("right1D1", "D1right1"): "right1D1 D1C1 C1C2 C2D2 D2D1 D1right1",
    ("right1D1", "D3right3"): "right1D1 D1D2 D2D3 D3right3",
    ("right1D1", "D3top3"): "right1D1 D1D2 D2D3 D3top3",
    ("right2D2", "A0bottom0"): "right2D2 D2C2 C2B2 B2A2 A2A1 A1A0 A0bottom0",
    ("right2D2", "A0left0"): "right2D2 D2C2 C2C1 C1C0 C0B0 B0A0 A0left0",
    ("right2D2", "A1left1"): "right2D2 D2C2 C2C1 C1B1 B1A1 A1left1",
    ("right2D2", "A2left2"): "right2D2 D2C2 C2B2 B2A2 A2left2",
    ("right2D2", "A3left3"): "right2D2 D2D3 D3C3 C3B3 B3A3 A3left3",
    ("right2D2", "A3top0"): "right2D2 D2C2 C2B2 B2A2 A2A3 A3top0",
    ("right2D2", "B0bottom1"): "right2D2 D2C2 C2B2 B2B1 B1B0 B0bottom1",
    ("right2D2", "B3top1"): "right2D2 D2C2 C2B2 B2B3 B3top1",
    ("right2D2", "C0bottom2"): "right2D2 D2C2 C2C1 C1C0 C0bottom2",
    ("right2D2", "C3top2"): "right2D2 D2C2 C2C3 C3top2",
    ("right2D2", "D0bottom3"): "right2D2 D2D1 D1D0 D0bottom3",
    ("right2D2", "D0right0"): "right2D2 D2D1 D1D0 D0right0",
    ("right2D2", "D1right1"): "right2D2 D2D1 D1right1",
    ("right2D2", "D2right2"): "right2D2 D2C2 C2C3 C3D3 D3D2 D2right2",
    ("right2D2", "D3right3"): "right2D2 D2D3 D3right3",
    ("right2D2", "D3top3"): "right2D2 D2D3 D3top3",
    ("right3D3", "A0bottom0"): "right3D3 D3C3 C3B3 B3A3 A3A2 A2A1 A1A0 A0bottom0",
    ("right3D3", "A0left0"): "right3D3 D3C3 C3C2 C2C1 C1C0 C0B0 B0A0 A0left0",
    ("right3D3", "A1left1"): "right3D3 D3C3 C3C2 C2C1 C1B1 B1A1 A1left1",
    ("right3D3", "A2left2"): "right3D3 D3C3 C3C2 C2B2 B2A2 A2left2",
    ("right3D3", "A3left3"): "right3D3 D3C3 C3B3 B3A3 A3left3",
    ("right3D3", "A3top0"): "right3D3 D3C3 C3B3 B3A3 A3top0",
    ("right3D3", "B0bottom1"): "right3D3 D3C3 C3B3 B3B2 B2B1 B1B0 B0bottom1",
    ("right3D3", "B3top1"): "right3D3 D3C3 C3B3 B3top1",
    ("right3D3", "C0bottom2"): "right3D3 D3C3 C3C2 C2C1 C1C0 C0bottom2",
    ("right3D3", "C3top2"): "right3D3 D3C3 C3top2",
    ("right3D3", "D0bottom3"): "right3D3 D3D2 D2D1 D1D0 D0bottom3",
    ("right3D3", "D0right0"): "right3D3 D3D2 D2D1 D1D0 D0right0",
    ("right3D3", "D1right1"): "right3D3 D3D2 D2D1 D1right1",
    ("right3D3", "D2right2"): "right3D3 D3D2 D2right2",
    ("right3D3", "D3right3"): "right3D3 D3D2 D2C2 C2C3 C3D3 D3right3",
    ("right3D3", "D3top3"): "right3D3 D3top3",
    ("top0A3", "A0bottom0"): "top0A3 A3A2 A2A1 A1A0 A0bottom0",
    ("top0A3", "A0left0"): "top0A3 A3A2 A2A1 A1A0 A0left0",
    ("top0A3", "A1left1"): "top0A3 A3A2 A2A1 A1left1",
    ("top0A3", "A2left2"): "top0A3 A3A2 A2left2",
    ("top0A3", "A3left3"): "top0A3 A3left3",
    ("top0A3", "A3top0"): "top0A3 A3B3 B3B2 B2A2 A2A3 A3top0",
    ("top0A3", "B0bottom1"): "top0A3 A3A2 A2B2 B2B1 B1B0 B0bottom1",
    ("top0A3", "B3top1"): "top0A3 A3B3 B3top1",
    ("top0A3", "C0bottom2"): "top0A3 A3A2 A2B2 B2C2 C2C1 C1C0 C0bottom2",
    ("top0A3", "C3top2"): "top0A3 A3B3 B3C3 C3top2",
    ("top0A3", "D0bottom3"): "top0A3 A3A2 A2B2 B2C2 C2D2 D2D1 D1D0 D0bottom3",
    ("top0A3", "D0right0"): "top0A3 A3A2 A2A1 A1A0 A0B0 B0C0 C0D0 D0right0",
    ("top0A3", "D1right1"): "top0A3 A3A2 A2A1 A1B1 B1C1 C1D1 D1right1",
    ("top0A3", "D2right2"): "top0A3 A3A2 A2B2 B2C2 C2D2 D2right2",
    ("top0A3", "D3right3"): "top0A3 A3B3 B3C3 C3D3 D3right3",
    ("top0A3", "D3top3"): "top0A3 A3B3 B3C3 C3D3 D3top3",
    ("top1B3", "A0bottom0"): "top1B3 B3A3 A3A2 A2A1 A1A0 A0bottom0",
    ("top1B3", "A0left0"): "top1B3 B3B2 B2B1 B1B0 B0A0 A0left0",
    ("top1B3", "A1left1"): "top1B3 B3B2 B2B1 B1A1 A1left1",
    ("top1B3", "A2left2"): "top1B3 B3B2 B2A2 A2left2",
    ("top1B3", "A3left3"): "top1B3 B3A3 A3left3",
    ("top1B3", "A3top0"): "top1B3 B3A3 A3top0",
    ("top1B3", "B0bottom1"): "top1B3 B3B2 B2B1 B1B0 B0bottom1",
    ("top1B3", "B3top1"): "top1B3 B3B2 B2A2 A2A3 A3B3 B3top1",
    ("top1B3", "C0bottom2"): "top1B3 B3B2 B2C2 C2C1 C1C0 C0bottom2",
    ("top1B3", "C3top2"): "top1B3 B3C3 C3top2",
    ("top1B3", "D0bottom3"): "top1B3 B3B2 B2C2 C2D2 D2D1 D1D0 D0bottom3",
    ("top1B3", "D0right0"): "top1B3 B3B2 B2B1 B1B0 B0C0 C0D0 D0right0",
    ("top1B3", "D1right1"): "top1B3 B3B2 B2B1 B1C1 C1D1 D1right1",
    ("top1B3", "D2right2"): "top1B3 B3B2 B2C2 C2D2 D2right2",
    ("top1B3", "D3right3"): "top1B3 B3C3 C3D3 D3right3",
    ("top1B3", "D3top3"): "top1B3 B3C3 C3D3 D3top3",
    ("top2C3", "A0bottom0"): "top2C3 C3C2 C2C1 C1C0 C0B0 B0A0 A0bottom0",
    ("top2C3", "A0left0"): "top2C3 C3C2 C2C1 C1C0 C0B0 B0A0 A0left0",
    ("top2C3", "A1left1"): "top2C3 C3C2 C2C1 C1B1 B1A1 A1left1",
    ("top2C3", "A2left2"): "top2C3 C3C2 C2B2 B2A2 A2left2",
    ("top2C3", "A3left3"): "top2C3 C3B3 B3A3 A3left3",
    ("top2C3", "A3top0"): "top2C3 C3B3 B3A3 A3top0",
    ("top2C3", "B0bottom1"): "top2C3 C3B3 B3B2 B2B1 B1B0 B0bottom1",
    ("top2C3", "B3top1"): "top2C3 C3B3 B3top1",
    ("top2C3", "C0bottom2"): "top2C3 C3C2 C2C1 C1C0 C0bottom2",
    ("top2C3", "C3top2"): "top2C3 C3C2 C2B2 B2B3 B3C3 C3top2",
    ("top2C3", "D0bottom3"): "top2C3 C3C2 C2D2 D2D1 D1D0 D0bottom3",
    ("top2C3", "D0right0"): "top2C3 C3C2 C2C1 C1C0 C0D0 D0right0",
    ("top2C3", "D1right1"): "top2C3 C3C2 C2C1 C1D1 D1right1",
    ("top2C3", "D2right2"): "top2C3 C3C2 C2D2 D2right2",
    ("top2C3", "D3right3"): "top2C3 C3D3 D3right3",
    ("top2C3", "D3top3"): "top2C3 C3D3 D3top3",
    ("top3D3", "A0bottom0"): "top3D3 D3D2 D2D1 D1D0 D0C0 C0B0 B0A0 A0bottom0",
    ("top3D3", "A0left0"): "top3D3 D3D2 D2D1 D1D0 D0C0 C0B0 B0A0 A0left0",
    ("top3D3", "A1left1"): "top3D3 D3D2 D2D1 D1C1 C1B1 B1A1 A1left1",
    ("top3D3", "A2left2"): "top3D3 D3D2 D2C2 C2B2 B2A2 A2left2",
    ("top3D3", "A3left3"): "top3D3 D3C3 C3B3 B3A3 A3left3",
    ("top3D3", "A3top0"): "top3D3 D3C3 C3B3 B3A3 A3top0",
    ("top3D3", "B0bottom1"): "top3D3 D3D2 D2D1 D1D0 D0C0 C0B0 B0bottom1",
    ("top3D3", "B3top1"): "top3D3 D3C3 C3B3 B3top1",
    ("top3D3", "C0bottom2"): "top3D3 D3C3 C3C2 C2C1 C1C0 C0bottom2",
    ("top3D3", "C3top2"): "top3D3 D3C3 C3top2",
    ("top3D3", "D0bottom3"): "top3D3 D3D2 D2D1 D1D0 D0bottom3",
    ("top3D3", "D0right0"): "top3D3 D3D2 D2D1 D1D0 D0right0",
    ("top3D3", "D1right1"): "top3D3 D3D2 D2D1 D1right1",
    ("top3D3", "D2right2"): "top3D3 D3D2 D2right2",
    ("top3D3", "D3right3"): "top3D3 D3right3",
    ("top3D3", "D3top3"): "top3D3 D3D2 D2C2 C2C3 C3D3 D3top3",
}

EXITS_BY_ENTRY=defaultdict(list)
for (entry,exit_) in ROUTE_MAP.keys(): EXITS_BY_ENTRY[entry].append(exit_)
ALL_ENTRIES=list(EXITS_BY_ENTRY.keys())
ENTRY_DIR={e:("WE" if e.startswith("left") else "EW" if e.startswith("right")
              else "SN" if e.startswith("bottom") else "NS") for e in ALL_ENTRIES}

SCENARIO_CFG={
    "original":{"base":0.021,"tf":lambda t:0.4 if t<720 else 1.8 if t<2700 else 0.7,"df":lambda d,t:1.0},
    "light":   {"base":0.007,"tf":lambda t:1.0,"df":lambda d,t:1.0},
    "medium":  {"base":0.021,"tf":lambda t:1.0,"df":lambda d,t:1.0},
    "heavy":   {"base":0.042,"tf":lambda t:1.0,"df":lambda d,t:1.0},
    "uneven":  {"base":0.012,"tf":lambda t:1.0,"df":lambda d,t:5.0 if d=="WE" else 1.0},
}

def generate_demand(scenario,episode,seed=None):
    cfg=SCENARIO_CFG[scenario]
    rng=np.random.default_rng(seed if seed is not None else 1000+episode)
    lines=['<?xml version="1.0" encoding="utf-8"?>','<routes>',
           '    <vType id="car" length="5.0" width="2.0" minGap="2.5" maxSpeed="11.111" accel="2.0" decel="4.5"/>']
    veh_id=0
    for t in range(SIM_STEPS):
        for entry in sorted(ALL_ENTRIES):
            prob=cfg["base"]*cfg["tf"](t)*cfg["df"](ENTRY_DIR[entry],t)
            if rng.random()<prob:
                exit_=EXITS_BY_ENTRY[entry][int(rng.integers(len(EXITS_BY_ENTRY[entry])))]
                lines.append(f'    <vehicle id="v{veh_id}" type="car" depart="{t}.00" '                             f'departLane="best" departSpeed="0"><route edges="{ROUTE_MAP[(entry,exit_)]}"/></vehicle>')
                veh_id+=1
    lines.append('</routes>')
    path=os.path.join(BASE_DIR,f"_dyn_h_{scenario}_ep{episode}.xml")
    with open(path,"w") as f: f.write("\n".join(lines))
    return path,veh_id

def pick_scenario(ep): return SCENARIOS[(ep-1)%len(SCENARIOS)]

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def _make_tls_phases():
    ew_g="G"*54+"r"*54; ew_y="y"*54+"r"*54; ns_g="r"*54+"G"*54; ns_y="r"*54+"y"*54
    return [traci.trafficlight.Phase(30,ew_g),traci.trafficlight.Phase(YELLOW_DUR,ew_y),
            traci.trafficlight.Phase(30,ns_g),traci.trafficlight.Phase(YELLOW_DUR,ns_y)]

def _lane_ok(l):
    try: traci.lane.getLastStepHaltingNumber(l); return True
    except: return False

class GridEnv:
    def __init__(self,gui=False,seed=42):
        self.gui=gui; self.seed=seed; self._live=False
        self._phase_idx={tls:0 for tls in TLS_IDS}; self._phase_dur={tls:0 for tls in TLS_IDS}; self._step=0
    def reset(self,route_file):
        if self._live:
            try: traci.close()
            except: pass
            time.sleep(0.5)
        cmd=["sumo-gui" if self.gui else "sumo","-n",NET_FILE,"-r",route_file,
             "--no-step-log","--no-warnings","--seed",str(self.seed),"--time-to-teleport","-1"]
        traci.start(cmd); self._live=True; self._step=0
        self._phase_idx={tls:0 for tls in TLS_IDS}; self._phase_dur={tls:0 for tls in TLS_IDS}
        for tls in TLS_IDS:
            try:
                logic=traci.trafficlight.Logic("rl",0,0,_make_tls_phases())
                traci.trafficlight.setProgramLogic(tls,logic); traci.trafficlight.setPhase(tls,GREEN_PHASES[0])
            except: pass
        return {tls:self._obs(tls) for tls in TLS_IDS}
    def step(self,actions):
        switching=[]
        for tls,action in actions.items():
            target=GREEN_PHASES[action]
            try: cur=traci.trafficlight.getPhase(tls)
            except: cur=target
            if cur!=target: switching.append((tls,cur+1,target))
            else: self._phase_dur[tls]+=DELTA_T
        if switching:
            for tls,yellow,_ in switching:
                try: traci.trafficlight.setPhase(tls,yellow)
                except: pass
            for _ in range(YELLOW_DUR): traci.simulationStep(); self._step+=1
            for tls,_,target in switching:
                try: traci.trafficlight.setPhase(tls,target)
                except: pass
                self._phase_idx[tls]=actions[tls]; self._phase_dur[tls]=0
        for tls,action in actions.items(): self._phase_idx[tls]=action
        for _ in range(DELTA_T): traci.simulationStep(); self._step+=1
        obs={tls:self._obs(tls) for tls in TLS_IDS}
        rews={tls:self._reward(tls) for tls in TLS_IDS}
        done=self._step>=SIM_STEPS
        pjq={tls:self._total_queue(tls) for tls in TLS_IDS}
        pjw={tls:self._avg_wait(tls) for tls in TLS_IDS}
        info={"global_reward":float(np.mean(list(rews.values()))),
              "avg_queue":float(np.mean(list(pjq.values()))),
              "avg_wait":float(np.mean(list(pjw.values()))),
              "throughput":traci.simulation.getArrivedNumber(),
              "per_junc_queue":pjq,"per_junc_wait":pjw}
        if done:
            try: traci.close()
            except: pass
            self._live=False
        return obs,rews,done,info
    def _queues(self,tls):
        q=[]
        for lane in AGENT_LANES[tls]:
            try: v=traci.lane.getLastStepHaltingNumber(lane)
            except: v=0
            q.append(min(v,MAX_QUEUE)/MAX_QUEUE)
        return np.array(q,dtype=np.float32)
    def _waits(self,tls):
        w=[]
        for lane in AGENT_LANES[tls]:
            try: v=traci.lane.getWaitingTime(lane)
            except: v=0.0
            w.append(min(v,120.0)/120.0)
        return np.array(w,dtype=np.float32)
    def _obs(self,tls):
        ph=np.zeros(NUM_PHASES,dtype=np.float32); ph[self._phase_idx[tls]]=1.0
        dur=np.array([min(self._phase_dur[tls],MAX_GREEN)/MAX_GREEN],dtype=np.float32)
        return np.concatenate([self._queues(tls),self._waits(tls),ph,dur])
    def _total_queue(self,tls):
        return float(sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENT_LANES[tls] if _lane_ok(l)))
    def _avg_wait(self,tls):
        w=[traci.lane.getWaitingTime(l) for l in AGENT_LANES[tls] if _lane_ok(l)]
        return float(np.mean(w)) if w else 0.0
    def _reward(self,tls):
        return -(self._total_queue(tls)+0.1*self._avg_wait(tls)/60.0)
    def close(self):
        if self._live:
            try: traci.close()
            except: pass
            self._live=False

# ══════════════════════════════════════════════════════════════════════════════
# 4. NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

class _PPOActor(nn.Module):
    def __init__(self,in_dim,act_dim,h=128):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(in_dim,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.pi=nn.Linear(h,act_dim); nn.init.orthogonal_(self.pi.weight,0.01)
    def forward(self,x): return self.pi(self.body(x))
    @torch.no_grad()
    def act_greedy(self,x_np): return int(self.forward(torch.FloatTensor(x_np).unsqueeze(0)).argmax())
    def act_stoch(self,x_np):
        x=torch.FloatTensor(x_np).unsqueeze(0); logits=self.forward(x)
        dist=Categorical(logits=logits); a=dist.sample()
        return a.item(),dist.log_prob(a).item()
    def evaluate(self,x_t,a_t):
        dist=Categorical(logits=self.forward(x_t)); return dist.log_prob(a_t),dist.entropy()

class _PPOCritic(nn.Module):
    def __init__(self,in_dim,h=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
        nn.init.orthogonal_(self.net[-1].weight,1.0)
    def forward(self,x): return self.net(x).squeeze(-1)

class _DuelingQNet(nn.Module):
    def __init__(self,in_dim,act_dim,h=128):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.val=nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,1))
        self.adv=nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,act_dim))
    def forward(self,x):
        f=self.body(x); v=self.val(f); a=self.adv(f)
        return v+a-a.mean(dim=-1,keepdim=True)

class ReplayBuffer:
    def __init__(self,cap): self.buf=deque(maxlen=cap)
    def push(self,o,a,r,no,d): self.buf.append((o,a,r,no,float(d)))
    def sample(self,n):
        b=random.sample(self.buf,n); o,a,r,no,d=zip(*b)
        return (np.array(o,dtype=np.float32),np.array(a,dtype=np.int64),
                np.array(r,dtype=np.float32),np.array(no,dtype=np.float32),
                np.array(d,dtype=np.float32))
    def __len__(self): return len(self.buf)

# ══════════════════════════════════════════════════════════════════════════════
# 5. H-PPO AGENT
# ══════════════════════════════════════════════════════════════════════════════

class HPPOAgent:
    ROLLOUT=128; EPOCHS=6; MB=64; GAMMA=0.98; LAM=0.95
    CLIP=0.2; LR=5e-4; ENT_C=0.05; GRAD_CLIP=1.0

    def __init__(self):
        self._rew_mean=0.0; self._rew_var=1.0; self._rew_count=0
        self.global_actor  =_PPOActor(GLOBAL_OBS_DIM,NUM_REGIONS,h=256)
        self.global_critic =_PPOCritic(GLOBAL_OBS_DIM,h=256)
        self.regional_actor=_PPOActor(REGIONAL_INPUT_DIM,4,h=128)
        self.regional_critic=_PPOCritic(REGIONAL_INPUT_DIM,h=128)
        self.local_actor   =_PPOActor(LOCAL_INPUT_DIM,NUM_PHASES,h=128)
        self.local_critic  =_PPOCritic(LOCAL_INPUT_DIM,h=128)
        self.opt_ga=optim.Adam(list(self.global_actor.parameters())+list(self.global_critic.parameters()),lr=self.LR,eps=1e-5)
        self.opt_ra=optim.Adam(list(self.regional_actor.parameters())+list(self.regional_critic.parameters()),lr=self.LR,eps=1e-5)
        self.opt_la=optim.Adam(list(self.local_actor.parameters())+list(self.local_critic.parameters()),lr=self.LR,eps=1e-5)
        self._reset_bufs()

    def _reset_bufs(self):
        self.g_obs=[]; self.g_act=[]; self.g_rew=[]; self.g_val=[]; self.g_lp=[]; self.g_don=[]
        self.r_obs=[]; self.r_act=[]; self.r_rew=[]; self.r_val=[]; self.r_lp=[]; self.r_don=[]
        self.l_obs=[]; self.l_act=[]; self.l_rew=[]; self.l_val=[]; self.l_lp=[]; self.l_don=[]

    def _diroh(self,d): oh=np.zeros(DIRECTIVE_DIM,dtype=np.float32); oh[d%DIRECTIVE_DIM]=1.0; return oh

    def act(self,obs_dict):
        glob=np.concatenate([obs_dict[tls] for tls in TLS_IDS])
        g_act,g_lp=self.global_actor.act_stoch(glob)
        with torch.no_grad(): g_val=self.global_critic(torch.FloatTensor(glob).unsqueeze(0)).item()
        g_dir=self._diroh(g_act)
        r_acts={};r_lps={};r_vals={};r_inps={};reg_dirs={}
        for reg in REGION_IDS:
            members=REGIONS[reg]; ro=np.concatenate([obs_dict[t] for t in members])
            ri=np.concatenate([ro,g_dir]); r_inps[reg]=ri
            ra,rl=self.regional_actor.act_stoch(ri)
            with torch.no_grad(): rv=self.regional_critic(torch.FloatTensor(ri).unsqueeze(0)).item()
            r_acts[reg]=ra; r_lps[reg]=rl; r_vals[reg]=rv
            rd=self._diroh(ra)
            for t in members: reg_dirs[t]=rd
        actions={}; l_lps={}; l_vals={}; l_inps={}
        for tls in TLS_IDS:
            li=np.concatenate([obs_dict[tls],reg_dirs[tls]]); l_inps[tls]=li
            la,ll=self.local_actor.act_stoch(li)
            with torch.no_grad(): lv=self.local_critic(torch.FloatTensor(li).unsqueeze(0)).item()
            actions[tls]=la; l_lps[tls]=ll; l_vals[tls]=lv
        return actions,glob,g_act,g_lp,g_val,r_inps,r_acts,r_lps,r_vals,l_inps,l_lps,l_vals

    def _norm_rew(self,rews):
        self._rew_count+=len(rews); bm=np.mean(rews)
        delta=bm-self._rew_mean
        self._rew_mean+=delta*len(rews)/self._rew_count
        self._rew_var=(self._rew_var*(self._rew_count-len(rews))+np.var(rews)*len(rews))/self._rew_count
        return (np.array(rews,dtype=np.float32)-self._rew_mean)/(np.sqrt(self._rew_var+1e-8))

    def remember(self,rewards,done,glob,g_act,g_lp,g_val,r_inps,r_acts,r_lps,r_vals,l_inps,l_lps,l_vals,actions):
        gr=float(np.mean(list(rewards.values())))
        self.g_obs.append(glob); self.g_act.append(g_act); self.g_rew.append(gr)
        self.g_val.append(g_val); self.g_lp.append(g_lp); self.g_don.append(float(done))
        for reg in REGION_IDS:
            members=REGIONS[reg]; rr=float(np.mean([rewards[t] for t in members]))
            self.r_obs.append(r_inps[reg]); self.r_act.append(r_acts[reg]); self.r_rew.append(rr)
            self.r_val.append(r_vals[reg]); self.r_lp.append(r_lps[reg]); self.r_don.append(float(done))
        for tls in TLS_IDS:
            self.l_obs.append(l_inps[tls]); self.l_act.append(actions[tls]); self.l_rew.append(rewards[tls])
            self.l_val.append(l_vals[tls]); self.l_lp.append(l_lps[tls]); self.l_don.append(float(done))

    def ready(self): return len(self.g_obs)>=self.ROLLOUT

    def _ppo_update(self,obs_l,act_l,rew_l,val_l,lp_l,don_l,actor,critic,opt,lv=0.0):
        oa=np.array(obs_l,dtype=np.float32); aa=np.array(act_l,dtype=np.int64)
        ra=self._norm_rew(rew_l); va=np.array(val_l,dtype=np.float32)
        la=np.array(lp_l,dtype=np.float32); da=np.array(don_l,dtype=np.float32); n=len(oa)
        adv=np.zeros(n,dtype=np.float32); gae=0.0
        for t in reversed(range(n)):
            nv=lv if t==n-1 else va[t+1]
            d=ra[t]+self.GAMMA*nv*(1-da[t])-va[t]
            gae=d+self.GAMMA*self.LAM*(1-da[t])*gae; adv[t]=gae
        ret=adv+va; adv=(adv-adv.mean())/(adv.std()+1e-8)
        ot=torch.FloatTensor(oa); at=torch.LongTensor(aa)
        olt=torch.FloatTensor(la); adt=torch.FloatTensor(adv); ret_t=torch.FloatTensor(ret)
        total=0.0
        for _ in range(self.EPOCHS):
            idx=torch.randperm(n)
            for s in range(0,n,self.MB):
                b=idx[s:s+self.MB]; nlp,ent=actor.evaluate(ot[b],at[b])
                ratio=torch.exp(nlp-olt[b])
                s1=ratio*adt[b]; s2=torch.clamp(ratio,1-self.CLIP,1+self.CLIP)*adt[b]
                pl=-torch.min(s1,s2).mean()-self.ENT_C*ent.mean()
                vl=nn.functional.mse_loss(critic(ot[b]),ret_t[b])
                loss=pl+0.5*vl; opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),self.GRAD_CLIP)
                opt.step(); total+=loss.item()
        return total/max(self.EPOCHS*max(n//self.MB,1),1)

    def learn(self,lg=0.0,lr=0.0,ll=0.0):
        gl=self._ppo_update(self.g_obs,self.g_act,self.g_rew,self.g_val,self.g_lp,self.g_don,self.global_actor,self.global_critic,self.opt_ga,lg)
        rl=self._ppo_update(self.r_obs,self.r_act,self.r_rew,self.r_val,self.r_lp,self.r_don,self.regional_actor,self.regional_critic,self.opt_ra,lr)
        ll2=self._ppo_update(self.l_obs,self.l_act,self.l_rew,self.l_val,self.l_lp,self.l_don,self.local_actor,self.local_critic,self.opt_la,ll)
        self._reset_bufs(); return gl,rl,ll2

    def end_episode(self): pass

    def save(self,p):
        torch.save({"global_actor":self.global_actor.state_dict(),"global_critic":self.global_critic.state_dict(),
                    "regional_actor":self.regional_actor.state_dict(),"regional_critic":self.regional_critic.state_dict(),
                    "local_actor":self.local_actor.state_dict(),"local_critic":self.local_critic.state_dict()},p)

# ══════════════════════════════════════════════════════════════════════════════
# 6. H-DQN AGENT
# ══════════════════════════════════════════════════════════════════════════════

class HDQNAgent:
    GAMMA=0.99; LR=5e-4; BATCH=256; TGT_SYNC=10
    G_BUF=50_000;  G_EPS_S=1.0; G_EPS_E=0.10; G_EPS_DEC=0.985
    R_BUF=100_000; R_EPS_S=1.0; R_EPS_E=0.08; R_EPS_DEC=0.988
    L_BUF=200_000; L_EPS_S=1.0; L_EPS_E=0.05; L_EPS_DEC=0.992

    def __init__(self):
        self.g_q=_DuelingQNet(GLOBAL_OBS_DIM,NUM_REGIONS,h=256)
        self.g_tgt=_DuelingQNet(GLOBAL_OBS_DIM,NUM_REGIONS,h=256)
        self.g_tgt.load_state_dict(self.g_q.state_dict()); self.g_tgt.eval()
        self.g_opt=optim.Adam(self.g_q.parameters(),lr=self.LR)
        self.g_buf=ReplayBuffer(self.G_BUF); self.g_eps=self.G_EPS_S

        self.r_q=_DuelingQNet(REGIONAL_INPUT_DIM,4,h=128)
        self.r_tgt=_DuelingQNet(REGIONAL_INPUT_DIM,4,h=128)
        self.r_tgt.load_state_dict(self.r_q.state_dict()); self.r_tgt.eval()
        self.r_opt=optim.Adam(self.r_q.parameters(),lr=self.LR)
        self.r_buf=ReplayBuffer(self.R_BUF); self.r_eps=self.R_EPS_S

        self.l_q=_DuelingQNet(LOCAL_INPUT_DIM,NUM_PHASES,h=128)
        self.l_tgt=_DuelingQNet(LOCAL_INPUT_DIM,NUM_PHASES,h=128)
        self.l_tgt.load_state_dict(self.l_q.state_dict()); self.l_tgt.eval()
        self.l_opt=optim.Adam(self.l_q.parameters(),lr=self.LR)
        self.l_buf=ReplayBuffer(self.L_BUF); self.l_eps=self.L_EPS_S
        self.loss_fn=nn.SmoothL1Loss(); self._ep=0

    def _diroh(self,d): oh=np.zeros(DIRECTIVE_DIM,dtype=np.float32); oh[d%DIRECTIVE_DIM]=1.0; return oh

    def act(self,obs_dict):
        glob=np.concatenate([obs_dict[tls] for tls in TLS_IDS])
        g_act=(random.randrange(NUM_REGIONS) if random.random()<self.g_eps
               else int(self.g_q(torch.FloatTensor(glob).unsqueeze(0)).argmax()))
        g_dir=self._diroh(g_act)
        r_acts={};r_inps={};reg_dirs={}
        for reg in REGION_IDS:
            members=REGIONS[reg]; ro=np.concatenate([obs_dict[t] for t in members])
            ri=np.concatenate([ro,g_dir]); r_inps[reg]=ri
            ra=(random.randrange(4) if random.random()<self.r_eps
                else int(self.r_q(torch.FloatTensor(ri).unsqueeze(0)).argmax()))
            r_acts[reg]=ra; rd=self._diroh(ra)
            for t in members: reg_dirs[t]=rd
        actions={};l_inps={}
        for tls in TLS_IDS:
            li=np.concatenate([obs_dict[tls],reg_dirs[tls]]); l_inps[tls]=li
            la=(random.randrange(NUM_PHASES) if random.random()<self.l_eps
                else int(self.l_q(torch.FloatTensor(li).unsqueeze(0)).argmax()))
            actions[tls]=la
        return actions,glob,g_act,r_inps,r_acts,l_inps

    def remember(self,rewards,done,glob,g_act,r_inps,r_acts,l_inps,actions,next_obs_dict):
        gr=float(np.mean(list(rewards.values())))
        ng=np.concatenate([next_obs_dict[tls] for tls in TLS_IDS])
        self.g_buf.push(glob,g_act,gr,ng,done)
        for reg in REGION_IDS:
            members=REGIONS[reg]; rr=float(np.mean([rewards[t] for t in members]))
            nro=np.concatenate([next_obs_dict[t] for t in members])
            nri=np.concatenate([nro,r_inps[reg][-DIRECTIVE_DIM:]])
            self.r_buf.push(r_inps[reg],r_acts[reg],rr,nri,done)
        for tls in TLS_IDS:
            nli=np.concatenate([next_obs_dict[tls],l_inps[tls][-DIRECTIVE_DIM:]])
            self.l_buf.push(l_inps[tls],actions[tls],rewards[tls],nli,done)

    def _dqn_update(self,buf,q,tgt,opt):
        if len(buf)<self.BATCH: return 0.0
        o,a,r,no,d=buf.sample(self.BATCH)
        ot=torch.FloatTensor(o); at=torch.LongTensor(a)
        rt=torch.FloatTensor(r); not_=torch.FloatTensor(no); dt=torch.FloatTensor(d)
        qv=q(ot).gather(1,at.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            ba=q(not_).argmax(1); nq=tgt(not_).gather(1,ba.unsqueeze(1)).squeeze(1)
            tgt_v=rt+self.GAMMA*nq*(1-dt)
        loss=self.loss_fn(qv,tgt_v)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(),10.0); opt.step()
        return loss.item()

    def learn(self):
        gl=self._dqn_update(self.g_buf,self.g_q,self.g_tgt,self.g_opt)
        rl=self._dqn_update(self.r_buf,self.r_q,self.r_tgt,self.r_opt)
        ll=self._dqn_update(self.l_buf,self.l_q,self.l_tgt,self.l_opt)
        return gl,rl,ll

    def end_episode(self):
        self._ep+=1
        self.g_eps=max(self.G_EPS_E,self.g_eps*self.G_EPS_DEC)
        self.r_eps=max(self.R_EPS_E,self.r_eps*self.R_EPS_DEC)
        self.l_eps=max(self.L_EPS_E,self.l_eps*self.L_EPS_DEC)
        if self._ep%self.TGT_SYNC==0:
            self.g_tgt.load_state_dict(self.g_q.state_dict())
            self.r_tgt.load_state_dict(self.r_q.state_dict())
            self.l_tgt.load_state_dict(self.l_q.state_dict())

    def save(self,p):
        torch.save({"g_q":self.g_q.state_dict(),"g_eps":self.g_eps,
                    "r_q":self.r_q.state_dict(),"r_eps":self.r_eps,
                    "l_q":self.l_q.state_dict(),"l_eps":self.l_eps},p)

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train(algo, episodes=300, gui=False, seed=42, use_wandb=True):
    import wandb

    algo_name={"hppo":"H-PPO","hdqn":"H-DQN"}[algo]
    if use_wandb:
        wandb.init(
            project="hierarchical-grid4x4",
            entity="TrafficLight_RL",
            name=f"{algo}-{episodes}ep",
            group=algo_name,
            config={"algo":algo,"episodes":episodes,"grid":"4x4","hierarchy":3,
                    "n_agents":NUM_AGENTS,"sim_steps":SIM_STEPS,"dynamic_demand":True})

    env=GridEnv(gui=gui,seed=seed)
    agent=HPPOAgent() if algo=="hppo" else HDQNAgent()
    history=[]; jw={tls:[] for tls in TLS_IDS}; jq={tls:[] for tls in TLS_IDS}; sl=[]
    max_steps=SIM_STEPS//DELTA_T

    print("="*65)
    print(f"  Algorithm : {algo_name} | Levels: Global→Regional→Local")
    print(f"  Episodes  : {episodes} | Dynamic demand: YES")
    print(f"  W&B       : {'enabled' if use_wandb else 'disabled'}")
    print("="*65)

    for ep in range(1,episodes+1):
        scen=pick_scenario(ep)
        route_file,n_veh=generate_demand(scen,ep,seed=seed+ep)
        print(f"\nEp {ep:4d}/{episodes} [{scen:<8}] {n_veh} veh",flush=True)
        obs_dict=env.reset(route_file); sl.append(scen)
        done=False; ep_r=ep_q=ep_w=0.0; g_l=r_l=l_l=0.0
        ep_jq={tls:0.0 for tls in TLS_IDS}; ep_jw={tls:0.0 for tls in TLS_IDS}
        steps=0; n_upd=0; lg=lr=ll=0.0

        while not done:
            if algo=="hppo":
                (actions,glob,g_act,g_lp,g_val,
                 r_inps,r_acts,r_lps,r_vals,
                 l_inps,l_lps,l_vals)=agent.act(obs_dict)
                next_obs,rewards,done,info=env.step(actions)
                agent.remember(rewards,done,glob,g_act,g_lp,g_val,
                               r_inps,r_acts,r_lps,r_vals,l_inps,l_lps,l_vals,actions)
                loss_g=loss_r=loss_l=0.0
                if agent.ready() and not done:
                    ng=np.concatenate([next_obs[tls] for tls in TLS_IDS])
                    with torch.no_grad():
                        nlg=agent.global_critic(torch.FloatTensor(ng).unsqueeze(0)).item()
                    loss_g,loss_r,loss_l=agent.learn(nlg,nlg,nlg); n_upd+=1
            else:
                (actions,glob,g_act,r_inps,r_acts,l_inps)=agent.act(obs_dict)
                next_obs,rewards,done,info=env.step(actions)
                agent.remember(rewards,done,glob,g_act,r_inps,r_acts,l_inps,actions,next_obs)
                loss_g,loss_r,loss_l=agent.learn()

            ep_r+=info["global_reward"]; ep_q+=info["avg_queue"]; ep_w+=info["avg_wait"]
            g_l+=loss_g; r_l+=loss_r; l_l+=loss_l; steps+=1
            for tls in TLS_IDS:
                ep_jq[tls]+=info["per_junc_queue"][tls]; ep_jw[tls]+=info["per_junc_wait"][tls]
            obs_dict=next_obs

            if steps%10==0 or done:
                pct=steps/max_steps*100
                bar="█"*(steps*20//max_steps)+"░"*(20-steps*20//max_steps)
                eps_str=""
                if algo=="hdqn": eps_str=f"|ε:{agent.l_eps:.2f}"
                print(f"\r  [{bar}]{pct:5.1f}%|Q:{ep_q/steps:.2f}|W:{ep_w/steps:.1f}s{eps_str}",
                      end="",flush=True)

        if algo=="hppo":
            loss_g,loss_r,loss_l=agent.learn(0.0,0.0,0.0)
        agent.end_episode()

        if os.path.exists(route_file): os.remove(route_file)
        s=max(steps,1)
        rec={"episode":ep,"scenario":scen,"reward":ep_r,
             "avg_queue":ep_q/s,"avg_wait":ep_w/s,
             "g_loss":g_l/s,"r_loss":r_l/s,"l_loss":l_l/s,
             "per_junc_queue":{t:v/s for t,v in ep_jq.items()},
             "per_junc_wait": {t:v/s for t,v in ep_jw.items()}}
        history.append(rec)
        for tls in TLS_IDS:
            jw[tls].append(rec["per_junc_wait"][tls])
            jq[tls].append(rec["per_junc_queue"][tls])

        print(f"\n  ✔ Ep {ep:4d} [{scen:<8}] R:{ep_r:.1f}|Q:{ep_q/s:.3f}|W:{ep_w/s:.2f}s|"
              f"Loss G/R/L:{g_l/s:.3f}/{r_l/s:.3f}/{l_l/s:.3f}",flush=True)

        if use_wandb:
            log={"episode":ep,"scenario":scen,"reward":ep_r,
                 "avg_queue":ep_q/s,"avg_wait":ep_w/s,
                 "loss/global":g_l/s,"loss/regional":r_l/s,"loss/local":l_l/s}
            if algo=="hdqn":
                log.update({"epsilon/global":agent.g_eps,"epsilon/regional":agent.r_eps,"epsilon/local":agent.l_eps})
            for tls in TLS_IDS:
                log[f"junction/{tls}/wait"]=rec["per_junc_wait"][tls]
                log[f"junction/{tls}/queue"]=rec["per_junc_queue"][tls]
            for reg,members in REGIONS.items():
                log[f"region/{reg}/wait"]=np.mean([rec["per_junc_wait"][t] for t in members])
                log[f"region/{reg}/queue"]=np.mean([rec["per_junc_queue"][t] for t in members])
            wandb.log(log)

        if ep%50==0: agent.save(f"{algo}_ep{ep}.pt")

    agent.save(f"{algo}_final.pt")
    np.save(f"{algo}_history.npy",history)
    np.save(f"{algo}_junc_wait.npy",jw)
    np.save(f"{algo}_junc_queue.npy",jq)
    np.save(f"{algo}_scenarios.npy",sl)
    if use_wandb: wandb.finish()
    print(f"\n✅ {algo_name} done.")
    return history,jw,jq,sl

# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(algo,history,jw,jq,sl):
    import matplotlib.pyplot as plt
    COLOR={"hppo":"#9b59b6","hdqn":"#e74c3c"}[algo]
    LABEL={"hppo":"H-PPO","hdqn":"H-DQN"}[algo]
    RCOLS={"R_A":"#e74c3c","R_B":"#2ecc71","R_C":"#3498db","R_D":"#f39c12"}
    SCEN_COLORS={"original":"#2c3e50","light":"#3498db","medium":"#2ecc71","heavy":"#e74c3c","uneven":"#f39c12"}
    eps=[h["episode"] for h in history]
    def smooth(x,w=10): return np.convolve(x,np.ones(w)/w,mode="valid") if len(x)>=w else np.array(x)

    # Fig 1: Training curves
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f"{LABEL} Training Curves — 4×4 Grid",fontsize=14,fontweight="bold")
    for ax,(key,title,note) in zip(axes,[("reward","Global Episode Reward","Higher ↑"),
        ("avg_queue","Avg Queue / Intersection","Lower ↓"),("avg_wait","Avg Wait Time (s)","Lower ↓")]):
        vals=[h[key] for h in history]; sm=smooth(vals)
        ax.plot(range(1,len(sm)+1),sm,color=COLOR,lw=2.5)
        ax.plot(eps,vals,color=COLOR,alpha=0.15,lw=1)
        ax.set_title(f"{title}\n{note}",fontsize=11); ax.set_xlabel("Episode"); ax.set_ylabel(title)
        ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(f"{algo}_training_curves.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_training_curves.png")

    # Fig 2: Loss per level
    fig2,axes2=plt.subplots(1,3,figsize=(18,4))
    fig2.suptitle(f"{LABEL} Loss Curves — All 3 Levels",fontsize=13,fontweight="bold")
    for ax,(key,title,color) in zip(axes2,[("g_loss","Global Agent Loss","#e74c3c"),
        ("r_loss","Regional Agent Loss","#2ecc71"),("l_loss","Local Agent Loss","#3498db")]):
        vals=[h[key] for h in history]; sm=smooth(vals)
        ax.plot(range(1,len(sm)+1),sm,color=color,lw=2)
        ax.plot(eps,vals,color=color,alpha=0.15,lw=1)
        ax.set_title(title,fontsize=11); ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
        ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig2.savefig(f"{algo}_loss_curves.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_loss_curves.png")

    # Fig 3: Final performance
    fig3,axes3=plt.subplots(1,3,figsize=(12,5))
    fig3.suptitle(f"{LABEL} Final Performance — Last 20 Episodes",fontsize=13,fontweight="bold")
    last20=history[-20:]
    for ax,(key,title,note) in zip(axes3,[("reward","Avg Global Reward","Higher ↑"),
        ("avg_queue","Avg Queue / Intersection","Lower ↓"),("avg_wait","Avg Wait Time (s)","Lower ↓")]):
        val=np.mean([h[key] for h in last20])
        bar=ax.bar([LABEL],[val],color=COLOR,width=0.4,edgecolor="white")
        ax.text(bar[0].get_x()+bar[0].get_width()/2,bar[0].get_height()*1.01,
                f"{val:.2f}",ha="center",va="bottom",fontsize=12,fontweight="bold")
        ax.set_title(f"{title}\n({note})",fontsize=11); ax.set_ylabel(title)
        ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig3.savefig(f"{algo}_final_performance.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_final_performance.png")

    # Fig 4: Per-scenario
    unique_scens=list(dict.fromkeys(sl))
    fig4,axes4=plt.subplots(1,3,figsize=(18,5))
    fig4.suptitle(f"{LABEL} Performance by Scenario",fontsize=13,fontweight="bold")
    for ax,(key,title,note) in zip(axes4,[("reward","Avg Reward","Higher ↑"),
        ("avg_queue","Avg Queue / Intersection","Lower ↓"),("avg_wait","Avg Wait Time (s)","Lower ↓")]):
        names=[]; vals=[]; colors=[]; stds=[]
        for scen in unique_scens:
            sv=[h[key] for h,s in zip(history,sl) if s==scen]
            if sv: names.append(scen); vals.append(np.mean(sv)); stds.append(np.std(sv)); colors.append(SCEN_COLORS.get(scen,"#95a5a6"))
        ax.bar(names,vals,color=colors,width=0.6,edgecolor="white",yerr=stds,capsize=4)
        ax.set_title(f"{title}\n({note})",fontsize=11); ax.set_ylabel(title)
        ax.tick_params(axis="x",rotation=30); ax.grid(axis="y",alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig4.savefig(f"{algo}_per_scenario.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_per_scenario.png")

    # Fig 5: Junction heatmap
    cols=["A","B","C","D"]
    fig5,axes5=plt.subplots(1,2,figsize=(14,5))
    fig5.suptitle(f"{LABEL} Per-Junction Performance — Last 20 Episodes",fontsize=13,fontweight="bold")
    for ax,(data,title) in zip(axes5,[(jw,"Avg Wait Time (s)"),(jq,"Avg Queue Length")]):
        grid=np.zeros((4,4))
        for ci,col in enumerate(cols):
            for row in range(4): grid[3-row,ci]=np.mean(data[f"{col}{row}"][-20:])
        im=ax.imshow(grid,cmap="YlOrRd",aspect="auto"); plt.colorbar(im,ax=ax)
        ax.set_xticks(range(4)); ax.set_xticklabels(cols)
        ax.set_yticks(range(4)); ax.set_yticklabels(["Row3","Row2","Row1","Row0"])
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f"{grid[i,j]:.1f}",ha="center",va="center",fontsize=9,
                        color="white" if grid[i,j]>grid.max()*0.6 else "black")
    plt.tight_layout(); fig5.savefig(f"{algo}_junction_heatmap.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_junction_heatmap.png")

    # Fig 6: Per-junction wait 4x4 grid
    fig6,axes6=plt.subplots(4,4,figsize=(20,14),sharex=True)
    fig6.suptitle(f"Wait Time per Junction — {LABEL}",fontsize=14,fontweight="bold")
    for ci,col in enumerate(cols):
        for row in range(4):
            tls=f"{col}{row}"; ax=axes6[3-row][ci]
            vals=jw[tls]; sm=smooth(vals)
            ax.plot(range(1,len(sm)+1),sm,color=COLOR,lw=1.5)
            ax.plot(eps,vals,color=COLOR,alpha=0.2,lw=0.8)
            ax.set_title(tls,fontsize=9,fontweight="bold")
            ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig6.savefig(f"{algo}_per_junction_wait.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_per_junction_wait.png")

    # Fig 7: Regional comparison
    fig7,axes7=plt.subplots(1,2,figsize=(14,5))
    fig7.suptitle(f"Regional Performance — {LABEL}",fontsize=13,fontweight="bold")
    for ax,(data,title,note) in zip(axes7,[(jw,"Avg Wait Time (s)","Lower ↓"),(jq,"Avg Queue Length","Lower ↓")]):
        for reg,members in REGIONS.items():
            rv=[np.mean([data[t][i] for t in members]) for i in range(len(eps))]
            sm=smooth(rv)
            ax.plot(range(1,len(sm)+1),sm,label=f"Region {reg[-1]} ({','.join(members)})",color=RCOLS[reg],lw=2)
        ax.set_title(f"{title} by Region\n{note}",fontsize=11); ax.set_xlabel("Episode"); ax.set_ylabel(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig7.savefig(f"{algo}_regional_comparison.png",dpi=150,bbox_inches="tight")
    print(f"✅ {algo}_regional_comparison.png")
    plt.close("all")   # auto-close, no manual intervention

def plot_compare():
    import matplotlib.pyplot as plt
    results={}
    for algo,label in [("hppo","H-PPO"),("hdqn","H-DQN")]:
        p=f"{algo}_history.npy"
        if os.path.exists(p): results[algo]={"label":label,"hist":np.load(p,allow_pickle=True).tolist()}
    if not results: print("No results found."); return
    COLORS={"hppo":"#9b59b6","hdqn":"#e74c3c"}
    def smooth(x,w=10): return np.convolve(x,np.ones(w)/w,mode="valid") if len(x)>=w else np.array(x)
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle("H-PPO vs H-DQN — Training Curves",fontsize=14,fontweight="bold")
    for ax,(key,title,note) in zip(axes,[("reward","Global Reward","Higher ↑"),
        ("avg_queue","Avg Queue / Intersection","Lower ↓"),("avg_wait","Avg Wait (s)","Lower ↓")]):
        for algo,d in results.items():
            vals=[h[key] for h in d["hist"]]; sm=smooth(vals); c=COLORS[algo]
            ax.plot(range(1,len(sm)+1),sm,color=c,lw=2.5,label=d["label"])
            ax.plot(range(1,len(vals)+1),vals,color=c,alpha=0.12,lw=1)
        ax.set_title(f"{title}\n{note}",fontsize=11); ax.set_xlabel("Episode"); ax.set_ylabel(title)
        ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig("hppo_vs_hdqn_training.png",dpi=150,bbox_inches="tight")
    fig2,axes2=plt.subplots(1,3,figsize=(14,5))
    fig2.suptitle("H-PPO vs H-DQN — Final Performance (Last 20 Episodes)",fontsize=13,fontweight="bold")
    for ax,(key,title,note) in zip(axes2,[("reward","Avg Global Reward","Higher ↑"),
        ("avg_queue","Avg Queue / Intersection","Lower ↓"),("avg_wait","Avg Wait (s)","Lower ↓")]):
        names=[d["label"] for d in results.values()]
        vals=[np.mean([h[key] for h in d["hist"][-20:]]) for d in results.values()]
        colors=[COLORS[a] for a in results.keys()]
        bars=ax.bar(names,vals,color=colors,width=0.4,edgecolor="white")
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,f"{v:.2f}",
                    ha="center",va="bottom",fontsize=12,fontweight="bold")
        ax.set_title(f"{title}\n({note})",fontsize=11); ax.set_ylabel(title)
        ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig2.savefig("hppo_vs_hdqn_final.png",dpi=150,bbox_inches="tight")
    print("✅ hppo_vs_hdqn_training.png | hppo_vs_hdqn_final.png")
    plt.close("all")   # auto-close, no manual intervention

# ══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Hierarchical MARL — 4×4 Grid")
    parser.add_argument("--algo",choices=["hppo","hdqn","all"],default="hppo")
    parser.add_argument("--episodes",type=int,default=300)
    parser.add_argument("--gui",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--no-wandb",action="store_true")
    parser.add_argument("--compare",action="store_true",help="Plot H-PPO vs H-DQN comparison")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()

    use_wandb=not args.no_wandb
    if use_wandb:
        try: import wandb
        except ImportError: print("pip install wandb"); use_wandb=False

    if args.compare: plot_compare(); sys.exit(0)

    ALGOS=["hppo","hdqn"] if args.algo=="all" else [args.algo]
    for algo in ALGOS:
        if args.plot_only:
            if os.path.exists(f"{algo}_history.npy"):
                hist=np.load(f"{algo}_history.npy",allow_pickle=True).tolist()
                jw=np.load(f"{algo}_junc_wait.npy",allow_pickle=True).item()
                jq=np.load(f"{algo}_junc_queue.npy",allow_pickle=True).item()
                sl=np.load(f"{algo}_scenarios.npy",allow_pickle=True).tolist() if os.path.exists(f"{algo}_scenarios.npy") else []
                plot_results(algo,hist,jw,jq,sl)
        else:
            hist,jw,jq,sl=train(algo,args.episodes,args.gui,args.seed,use_wandb)
            plot_results(algo,hist,jw,jq,sl)
    if len(ALGOS)==2: plot_compare()
