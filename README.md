# quactography

## Installing
Becoming a quactographer is easy. Just run the following command in the project root
```
pip install -e .
```

## Running scripts
Scripts are located in the `scripts` folder and are available from the command line (see example below).
```
build_graph.py --help
```

Toy data is included in the `data` folder. To build a graph, run the command below.
```
build_graph.py data/wm.nii.gz data/fodf.nii.gz graph.npz
```
See the `--help` output for additional options.

![quactrography](https://github.com/scilus/quactography/assets/2171665/cfd6da68-699f-4761-8e30-395d0d8930ec)

