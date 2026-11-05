"""Check the real local Gradio server on bounded MNIST inputs.

No trained user model is touched: all six registries are redirected to the
chosen validation directory for this process. Use --serve for a manual browser
check; the default writes HTTP/export evidence and shuts down automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/validation/ui")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import app
    from data_import import Dataset, load_mnist

    ds = load_mnist()
    bounded = Dataset(ds.X_train[:200], ds.y_train[:200], ds.X_test[:64], ds.y_test[:64],
                      ds.class_names, "mnist")
    for algorithm in app.ALGOS.values():
        registry = importlib.import_module(type(algorithm).__module__.rsplit(".", 1)[0] + ".utils.registry")
        registry.MODELS_DIR = output / "models" / algorithm.key
    pca = app.ALGOS["pca"]
    params = dict(n_components=8, center="feature", n_samples=200)
    weights, metadata = pca.train(bounded.X_train, params)
    metadata.update(algo="pca", dataset="mnist", n_samples=200)
    pca.save("verification_pca", weights, metadata)
    app.DEFAULT_ALGO = "pca"
    demo = app.build_app(initial_dataset=bounded)
    try:
        demo.queue(default_concurrency_limit=1).launch(
            server_name="127.0.0.1", server_port=args.port, inbrowser=False,
            share=False, prevent_thread_lock=True,
        )
        base = f"http://127.0.0.1:{args.port}"
        with urllib.request.urlopen(base, timeout=30) as response:
            status = response.status
            html = response.read()
        with urllib.request.urlopen(base + "/config", timeout=30) as response:
            config = json.load(response)
        gallery, archive_path, note = app.export_all(
            "pca", "mnist", "verification_pca", 32, 8, 42,
            progress=lambda *args, **kwargs: None,
        )
        # Keep the exported report alongside its evidence; the app's temporary
        # folder may be cleaned after the process exits.
        import shutil
        exported = output / "export_pca.zip"
        shutil.copyfile(archive_path, exported)
        with zipfile.ZipFile(exported) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            figure_count = len(manifest["figures"])
            assert archive.testzip() is None
        proof = {"http_status": status, "html_bytes": len(html),
                 "component_count": len(config["components"]),
                 "event_count": len(config["dependencies"]),
                 "dataset": "real MNIST, bounded to 200 train and 64 test",
                 "model": "PCA 8 components; validation artifact only",
                 "export_figure_count": figure_count,
                 "export_has_provenance_manifest": True,
                 "export_archive": str(exported.relative_to(ROOT)),
                 "url": base, "external_share": False}
        assert status == 200 and gallery and figure_count == len(gallery)
        (output / "http_export_check.json").write_text(
            json.dumps(proof, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(proof, ensure_ascii=True), flush=True)
        if args.serve:
            print("Browser verification ready. Interrupt this process to close the local server.", flush=True)
            import threading
            threading.Event().wait()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
