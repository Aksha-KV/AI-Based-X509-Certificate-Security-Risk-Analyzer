"""Minimal runner so the suite works even where pytest is not installed:  python -m tests.run_tests"""
import importlib, inspect, pkgutil, sys, time, traceback
import tests


def main() -> int:
    passed = failed = 0
    for mod in sorted(m.name for m in pkgutil.iter_modules(tests.__path__) if m.name.startswith("test_")):
        module = importlib.import_module(f"tests.{mod}")
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != module.__name__:
                continue
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"PASS {mod}.{name} ({time.time() - t0:.2f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {mod}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
