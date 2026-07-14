import os
import random
import torch
import platform
import subprocess
import datetime
import time
import sys

# ── Configuraciones Básicas ──
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(SCRIPT_DIR, 'CSV')
NATS_PATH = os.path.join(BASE_DIR, 'NATS-tss-v1_0-3ffb9-simple', 'NATS-tss-v1_0-3ffb9-simple')
CSV_GT = os.path.join(BASE_DIR, 'nats_ground_truth.csv')
CSV_ZCP = os.path.join(BASE_DIR, 'nats_zcps.csv')
CSV_ZCP_FULL = os.path.join(BASE_DIR, 'nats_zcps_full.csv')
CSV_ZCP_EXT = os.path.join(BASE_DIR, 'nats_zcps_extended.csv')
CIFAR_DIR = os.path.join(SCRIPT_DIR, 'tmp', 'cifar10')

DATASET, BUDGET = 'cifar10-valid', '200'

# ── Generación de Semilla (Reproducibilidad) ──
MASTER_SEED = random.randint(0, 2**31 - 1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Columnas de Zero-Cost Proxies ──
ZCP_COLS = ['grad_norm', 'snip', 'grasp', 'jacob_cov', 'synflow']
ZCP_COLS_WITH_PARAMS = ZCP_COLS + ['params_k']

# ── Espacio Extendido ──
CHANNELS = [8, 16, 32, 64]
NUM_CELLS = [3, 5, 7, 9]
SAMPLE_PER_GROUP = 50
OPS_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
N_EDGES = 6

# ── Generación de Carpetas de Salida y Timestamps ──
OUTPUTS = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUTS, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
GRAFICOS_DIR_BASE = os.path.join(OUTPUTS, f'graficos_{TIMESTAMP}')
GRAFICOS_DIR = GRAFICOS_DIR_BASE
contador = 1
while os.path.exists(GRAFICOS_DIR):
    GRAFICOS_DIR = f"{GRAFICOS_DIR_BASE}_{contador}"
    contador += 1

MODELOS_DIR = os.path.join(OUTPUTS, f'modelos_{TIMESTAMP}')

def setup_output_dirs():
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    os.makedirs(MODELOS_DIR, exist_ok=True)

METADATA_FILE = os.path.join(GRAFICOS_DIR, 'metadata.txt')

START_TIME_TOTAL = time.perf_counter()

# Intentar extraer hash del commit de Git si existe
try:
    GIT_COMMIT = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT, cwd=SCRIPT_DIR).decode('utf-8').strip()
except Exception:
    GIT_COMMIT = "No disponible"
