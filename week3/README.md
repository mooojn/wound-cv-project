# Week 3 - Object Detection

## Final Deliverables
- Trained model: `week3/runs/<run_name>/weights/best.pt`
- mAP metrics: `week3/results/map_results.csv`, `week3/results/map_results.json`
- Predictions: `week3/results/predictions/images`, `week3/results/predictions/labels`

## Commands
```powershell
python week3\scripts\prepare_detection_dataset.py
python week3\scripts\train_detection.py --device auto --name wound_detection_cpu
python week3\scripts\evaluate_detection.py --weights week3\runs\wound_detection_cpu\weights\best.pt --device auto
python week3\scripts\predict_detection.py --weights week3\runs\wound_detection_cpu\weights\best.pt --device auto
```

## Resume Training
```powershell
python -c "from ultralytics import YOLO; YOLO('week3/runs/wound_detection_cpu/weights/last.pt').train(resume=True, device=0)"
```
