Use CIFAR-10 and a small PyTorch CNN as the target classifier.
Defense transform: JPEG compression (PIL) + optional smoothing.
Detector: small MLP that takes the classifier's penultimate-layer features (or logits) and/or the L2 difference between original and transformed features and outputs benign/adversarial.
Adversarial generator for testing: FGSM (and optional PGD).
Flask backend: endpoints to upload image, run classifier with & without defense, run detector, simulate attacks, and log detections.
Simple web UI: show uploaded image, prediction (clean), prediction (defended), and a detection flag.
backend/train.py — trains classifier + detector on CIFAR-10 with adversarial examples (FGSM).
model.py — model definitions, defense transform, attack generation, feature extractor, inference pipeline.
app.py — Flask server with upload & simulate endpoints and logging.
requirements.txt — add ML deps.
Example code (place under /workspaces/ML-Manthon/backend)
