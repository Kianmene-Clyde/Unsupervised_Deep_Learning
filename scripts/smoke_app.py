"""Validate a real-data Gradio workflow and loopback HTTP, then stop the server.

This bounded check measures functionality, not model quality. Registries are
isolated, so existing experiment models are never changed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("KERAS_HOME", str(ROOT / "data" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "quickdraw"], default="mnist")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "validation" / "ui")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import app
    import gradio as gr
    from data_import import Dataset, load_dataset

    data = load_dataset(args.dataset)
    sample = Dataset(data.X_train[:64], data.y_train[:64], data.X_test[:32],
                     data.y_test[:32], data.class_names, data.name)
    evidence = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "functional_smoke_only_not_model_quality",
        "dataset": args.dataset, "synthetic": False,
        "source_split_sizes": {"train": len(data.X_train), "test": len(data.X_test)},
        "sample_sizes": {"train": len(sample.X_train), "test": len(sample.X_test)},
        "runtime": {"python": platform.python_version(), "gradio": gr.__version__},
        "checks": {},
    }
    registries = []
    demo = None
    with tempfile.TemporaryDirectory(prefix="registries_", dir=args.output) as temporary:
        try:
            for algo in app.ALGOS.values():
                registry = importlib.import_module(
                    type(algo).__module__.rsplit(".", 1)[0] + ".utils.registry")
                registries.append((registry, registry.MODELS_DIR))
                registry.MODELS_DIR = Path(temporary) / algo.key
            app._LOADED.clear()
            app._LOADED[args.dataset] = sample
            # UI's initial selector is MNIST; point its cached initial view at
            # the selected real dataset only during this isolated check.
            app.DEFAULT_DATASET = args.dataset
            demo = app.build_app(initial_dataset=sample)
            evidence["checks"]["build"] = "passed"

            values = []
            for key, name in app.PARAM_SLOTS:
                value = next(p.default for p in app.ALGOS[key].params if p.name == name)
                if key == "kmeans" and name == "k":
                    value = 2
                if key == "kmeans" and name == "max_iter":
                    value = 2
                values.append(value)
            update, _, _ = app.train("kmeans", args.dataset, 64, "ui_smoke", *values)
            model_name = update["value"]
            evidence["checks"]["train_and_reload"] = "passed"
            evidence["model_name"] = model_name

            gallery, archive, _ = app.export_all(
                "kmeans", args.dataset, model_name, 24, 3, 42,
                progress=lambda *a, **kw: None,
            )
            shutil.copy2(archive, args.output / "report.zip")
            with zipfile.ZipFile(archive) as bundle:
                manifest = json.loads(bundle.read("manifest.json"))
                evidence["export"] = {"figure_count": len(gallery),
                                      "manifest": manifest}
                figures_dir = args.output / "figures"
                figures_dir.mkdir(exist_ok=True)
                for figure in gallery:
                    shutil.copy2(figure, figures_dir / Path(figure).name)
            evidence["checks"]["codec_latent_dictionary_extras_curve_export"] = "passed"

            _, local_url, _ = demo.queue(default_concurrency_limit=1).launch(
                server_name="127.0.0.1", server_port=args.port,
                inbrowser=False, share=False, prevent_thread_lock=True,
                theme=gr.themes.Soft(), quiet=True,
            )
            for suffix in ("", "config"):
                with urllib.request.urlopen(local_url + suffix, timeout=15) as response:
                    payload = response.read()
                    evidence["checks"]["http_" + (suffix or "root")] = {
                        "status": response.status, "bytes": len(payload)}
                    if suffix:
                        config = json.loads(payload)
                        evidence["component_count"] = len(config["components"])
                        evidence["event_count"] = len(config["dependencies"])
            # Exercise an actual queued callback over HTTP as well as the
            # direct export workflow above.
            from gradio_client import Client

            client = Client(local_url, verbose=False)
            curve = client.predict("kmeans", args.dataset, model_name,
                                   api_name="/run_curve")
            if not isinstance(curve, tuple) or len(curve) != 2:
                raise RuntimeError("La réponse HTTP de la courbe est invalide.")
            evidence["checks"]["queued_curve_http"] = "passed"
            evidence["result"] = "passed"
        except Exception as exc:
            evidence["result"] = "failed"
            evidence["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if demo is not None:
                demo.close()
                evidence["server_shutdown"] = True
            for registry, original in registries:
                registry.MODELS_DIR = original
            (args.output / "evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
    print(json.dumps({"result": evidence["result"],
                      "evidence": str(args.output / "evidence.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
