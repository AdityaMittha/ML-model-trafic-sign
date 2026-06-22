import json
from pathlib import Path

f_path = Path("notebooks/Traffic_Sign_Detection_Colab.ipynb")
data = json.load(open(f_path, encoding='utf-8'))

cell_source = [
    "BEST_MODEL = str(OUTPUTS_DIR / 'training' / 'traffic_sign_yolo11n' / 'weights' / 'best.pt')\n",
    "FALLBACK_MODEL = str(PROJECT_ROOT / 'runs' / 'detect' / 'outputs' / 'training' / 'traffic_sign_yolo11n' / 'weights' / 'best.pt')\n",
    "\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "if not Path(BEST_MODEL).exists() and Path(FALLBACK_MODEL).exists():\n",
    "    Path(BEST_MODEL).parent.mkdir(parents=True, exist_ok=True)\n",
    "    shutil.copy2(FALLBACK_MODEL, BEST_MODEL)\n",
    "    print('Self-healed: copied best.pt from fallback path.')\n",
    "\n",
    "from src.evaluation.metrics import MetricsEvaluator\n",
    "\n",
    "evaluator = MetricsEvaluator(BEST_MODEL, str(DATA_DIR / 'processed' / 'dataset.yaml'))\n",
    "report = evaluator.generate_report(OUTPUTS_DIR / 'evaluation')\n",
    "\n",
    "for split in ['val_metrics', 'test_metrics']:\n",
    "    m = report[split]\n",
    "    print(f'\\n=== {split} ===')\n",
    "    print(f'  Precision: {m[\"precision\"]:.4f}')\n",
    "    print(f'  Recall:    {m[\"recall\"]:.4f}')\n",
    "    print(f'  F1:        {m[\"f1\"]:.4f}')\n",
    "    print(f'  mAP50:     {m[\"mAP50\"]:.4f}')\n",
    "    print(f'  mAP50-95:  {m[\"mAP50-95\"]:.4f}')"
]

data['cells'][17]['source'] = cell_source

with open(f_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Updated Cell 17 in notebook successfully!")
