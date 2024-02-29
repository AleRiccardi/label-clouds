# Label Clouds

## How to configure it

```bash
export LABEL_PATH=/home/ar/Projects/label-clouds/
export PYTHONPATH="${PYTHONPATH}:${LABEL_PATH}"
```

```bash
pyhton -m venv .env
pip install -r configs/requirements.txt
```

Finally, get import the data folder.


## How to run it


```bash
python examples/connections.py configs/parameters/data_08_14.yaml
```

