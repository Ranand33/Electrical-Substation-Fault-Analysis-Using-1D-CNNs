# Electrical Substation Fault Analysis Using 1D-CNNs

1D-CNN models for predicting electrical substation events using Power Communication Module
(PCM) data. Compares Simple 1D-CNN, 1D ResNet, and TCN architectures against
classical baselines (Logistic Regression, Random Forest, XGBoost) on a binary
classification.

## Structure
```
lib/ = Core modules: models, dataset, training loop, evaluation, plots
steps/ = Pipeline scripts (run in numeric order)
run_pipeline.sh = End-to-end pipeline runner
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
./run_pipeline.sh
```
or run individual steps. For example,
```bash
./steps/01_preprocess.py
```

## License
MIT - see [LICENSE](LICENSE).
