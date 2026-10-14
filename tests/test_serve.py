"""Mode 2, ``serve``: IRENE compatibility, CF output, products, concurrency, operations and exit codes."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from run_testdata import build_input

import mlcast_icb.serve as serve_module
from mlcast_icb import __version__
from mlcast_icb.bundle import create_bundle, mlcast_version
from mlcast_icb.errors import ConfigError
from mlcast_icb.serve import (
    KNOWN_VARIABLES,
    SERVE_VARIABLES,
    LoadedModel,
    ServeConfig,
    create_app,
    describe_config,
    load_model,
    main,
)
from mlcast_icb.upstream.convert_convgru_ensemble import convert

#: The IRENE-shaped tiny bundle: 6 input steps, 12 forecast steps, 10 members, noisy decoder.
INPUT_STEPS = 6
FORECAST_STEPS = 12
ENSEMBLE_SIZE = 10
SIZE = 8


@pytest.fixture(scope="module")
def bundle_dir(tiny_convgru_ensemble_checkpoint: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("serve_bundle")
    checkpoint_path, _ = convert(tiny_convgru_ensemble_checkpoint, root / "converted")
    return create_bundle(checkpoint_path, root / "model", name="irene-tiny", model_version="0.1")


@pytest.fixture(scope="module")
def config(bundle_dir: Path) -> ServeConfig:
    return ServeConfig(model_dir=str(bundle_dir), device="cpu")


@pytest.fixture(scope="module")
def loaded(config: ServeConfig) -> LoadedModel:
    return load_model(config)


def wait_until(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.01)


@pytest.fixture
def make_client(config: ServeConfig, loaded: LoadedModel) -> Iterator[Callable[..., TestClient]]:
    """Start an app (sharing the loaded model) with config overrides, wait for its warm-up, return its client."""
    clients: list[TestClient] = []

    def _make(wait_ready: bool = True, **overrides: Any) -> TestClient:
        app = create_app(ServeConfig(**{**config.__dict__, **overrides}), loaded)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        if wait_ready:
            wait_until(app.state.server.ready.is_set)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def nc3(ds: xr.Dataset) -> bytes:
    """``ds`` as netCDF3 bytes, the way IRENE's own test builds its upload."""
    return bytes(ds.to_netcdf(engine="scipy"))


def irene_input(nt: int = INPUT_STEPS, ny: int = SIZE, nx: int = SIZE, seed: int = 0, **attrs: Any) -> bytes:
    """An upload as IRENE's clients send it: a bare ``RR(time, y, x)`` in netCDF3, no coordinates, no CF attributes."""
    data = np.random.default_rng(seed).gamma(0.5, 2.0, size=(nt, ny, nx)).astype("float32")
    return xr.Dataset({"RR": xr.DataArray(data, dims=["time", "y", "x"], attrs=attrs)}).pipe(nc3)


def netcdf4_bytes(ds: xr.Dataset, tmp_path: Path) -> bytes:
    path = tmp_path / "upload.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    return path.read_bytes()


def open_response(content: bytes, **kwargs: Any) -> xr.Dataset:
    with xr.open_dataset(io.BytesIO(content), engine="h5netcdf", **kwargs) as ds:
        return ds.load()


def post(client: TestClient, content: bytes, params: str = "", **kwargs: Any):
    return client.post(
        f"/predict{params}", files={"file": ("input.nc", io.BytesIO(content), "application/x-netcdf")}, **kwargs
    )


# --- IRENE compatibility: IRENE's own tests/test_serve.py, replayed --------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(client: TestClient, loaded: LoadedModel) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    # IRENE's keys, all present
    for key in (
        "architecture",
        "input_channels",
        "num_blocks",
        "forecast_steps",
        "ensemble_size",
        "noisy_decoder",
        "loss_class",
        "device",
    ):
        assert key in data
    assert data["architecture"].endswith("ConvGruModel")
    assert data["num_blocks"] == 2
    assert data["input_channels"] == 1
    assert data["noisy_decoder"] is True
    assert data["forecast_steps"] == FORECAST_STEPS
    assert data["ensemble_size"] == ENSEMBLE_SIZE
    assert data["device"] == "cpu"
    # and ours
    assert data["model_sha256"] == loaded.manifest["model_sha256"]
    assert data["standard_names"] == ["rainfall_rate"]
    assert data["input_steps"] == INPUT_STEPS
    assert data["timestep"] == "PT5M"


def test_predict_returns_netcdf(client: TestClient) -> None:
    resp = post(client, irene_input(), "?forecast_steps=12&ensemble_size=10")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-netcdf"
    assert "X-Elapsed-Seconds" in resp.headers


def test_irene_response_layout(client: TestClient) -> None:
    resp = post(client, irene_input(), "?forecast_steps=3&ensemble_size=2")
    assert resp.status_code == 200
    assert resp.headers["content-disposition"] == "attachment; filename=predictions.nc"
    float(resp.headers["X-Elapsed-Seconds"])
    out = open_response(resp.content)
    assert list(out.data_vars) == ["precipitation_forecast"]
    assert list(out.coords) == []
    da = out["precipitation_forecast"]
    assert da.dims == ("ensemble_member", "forecast_step", "y", "x")
    assert da.shape == (2, 3, SIZE, SIZE)
    assert da.dtype == np.float32
    assert da.attrs == {"units": "mm/h", "long_name": "Ensemble precipitation forecast"}
    assert da.encoding["zlib"] is True or da.encoding.get("compression") == "gzip"
    assert out.attrs["forecast_steps"] == 3
    assert out.attrs["ensemble_size"] == 2
    float(out.attrs["elapsed_seconds"])
    assert out.attrs["model"] == "irene-tiny"
    assert np.isfinite(da.values).all() and (da.values >= 0).all()


def test_irene_defaults_are_the_models(client: TestClient) -> None:
    out = open_response(post(client, irene_input()).content)
    assert out["precipitation_forecast"].shape == (ENSEMBLE_SIZE, FORECAST_STEPS, SIZE, SIZE)
    assert out.attrs["forecast_steps"] == FORECAST_STEPS
    assert out.attrs["ensemble_size"] == ENSEMBLE_SIZE


@pytest.mark.parametrize(
    "params", ["?forecast_steps=0", "?forecast_steps=49", "?ensemble_size=0", "?ensemble_size=11", "?complevel=10"]
)
def test_irene_limits_are_enforced(client: TestClient, params: str) -> None:
    assert post(client, irene_input(), params).status_code == 422


def test_more_than_the_model_steps_are_allowed(client: TestClient) -> None:
    out = open_response(post(client, irene_input(), "?forecast_steps=48&ensemble_size=1").content)
    assert out["precipitation_forecast"].shape == (1, 48, SIZE, SIZE)


def test_variable_selects_the_input(client: TestClient) -> None:
    data = np.zeros((INPUT_STEPS, SIZE, SIZE), "float32")
    content = xr.Dataset({"rain": (("time", "y", "x"), data)}).pipe(nc3)
    assert post(client, content, "?variable=rain").status_code == 200
    resp = post(client, content)
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("Variable 'RR' not found")
    resp = post(client, content, "?variable=nope")
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Variable 'nope' not found. Available: ['rain']"


def test_missing_file_is_422(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422
    resp = client.post("/predict", files={"other": ("x.nc", io.BytesIO(irene_input()), "application/x-netcdf")})
    assert resp.status_code == 422
    assert "No file uploaded" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"this is not netcdf", "not a netCDF file"),
        (
            xr.Dataset({"RR": (("y", "x"), np.zeros((8, 8), "float32"))}).pipe(nc3),
            "Expected 3D variable",
        ),
        (
            xr.Dataset({"RR": (("a", "b", "c"), np.zeros((8, 8, 6), "float32"))}).pipe(nc3),
            "First dimension should be time",
        ),
        (irene_input(nt=4), "needs at least 6 input time steps, got 4"),
        (
            xr.Dataset({"RR": (("time", "y", "x"), np.full((6, 8, 8), np.inf, "float32"))}).pipe(nc3),
            "Inf values",
        ),
    ],
    ids=["empty", "not-netcdf", "2d", "time-not-first", "too-few-times", "inf"],
)
def test_bad_inputs_are_422(client: TestClient, content: bytes, message: str) -> None:
    resp = post(client, content)
    assert resp.status_code == 422
    assert message in resp.json()["detail"]


def test_upload_over_the_limit_is_413(make_client: Callable[..., TestClient]) -> None:
    client = make_client(max_upload_mb=1)
    big = irene_input(ny=400, nx=400)  # 6 * 400 * 400 * 4 bytes, ~3.7 MiB
    resp = post(client, big)
    assert resp.status_code == 413
    assert "Maximum is 1 MB" in resp.json()["detail"]
    resp = client.put("/predict", content=big, headers={"Content-Type": "application/x-netcdf"})
    assert resp.status_code == 413


def test_nan_is_no_rain_and_mask_restores_it(client: TestClient) -> None:
    data = np.random.default_rng(0).gamma(0.5, 2.0, size=(INPUT_STEPS, SIZE, SIZE)).astype("float32")
    data[:, :2, :] = np.nan  # outside coverage at every time
    data[0, 4, 4] = np.nan  # missing once only
    content = xr.Dataset({"RR": (("time", "y", "x"), data)}).pipe(nc3)

    plain = open_response(post(client, content, "?seed=1&ensemble_size=2").content)["precipitation_forecast"]
    assert np.isfinite(plain.values).all()

    masked = open_response(post(client, content, "?seed=1&ensemble_size=2&mask=true").content)
    values = masked["precipitation_forecast"].values
    assert np.isnan(values[:, :, :2, :]).all()
    assert np.isfinite(values[:, :, 2:, :]).all()
    np.testing.assert_array_equal(values[:, :, 2:, :], plain.values[:, :, 2:, :])
    assert "mask" in masked.attrs


def test_irene_format_reads_values_as_mm_per_hour_whatever_the_units(client: TestClient) -> None:
    plain = post(client, irene_input(), "?seed=3&ensemble_size=2").content
    labelled = post(client, irene_input(units="kg m-2 s-1"), "?seed=3&ensemble_size=2").content
    np.testing.assert_array_equal(
        open_response(plain)["precipitation_forecast"].values, open_response(labelled)["precipitation_forecast"].values
    )


def test_only_the_last_input_steps_are_used(client: TestClient) -> None:
    six = irene_input(nt=INPUT_STEPS, seed=5)
    longer = xr.open_dataset(io.BytesIO(six), engine="scipy").load()
    padded = xr.concat([longer.isel(time=[0, 1, 2]) * 100, longer], dim="time").pipe(nc3)
    a = open_response(post(client, six, "?seed=0&ensemble_size=2").content)
    b = open_response(post(client, padded, "?seed=0&ensemble_size=2").content)
    np.testing.assert_array_equal(a["precipitation_forecast"].values, b["precipitation_forecast"].values)


# --- CF output -------------------------------------------------------------------------


def test_cf_format_schema(client: TestClient, tmp_path: Path, loaded: LoadedModel) -> None:
    ds = build_input(nt=8)
    resp = post(client, netcdf4_bytes(ds, tmp_path), "?format=cf&forecast_steps=3&ensemble_size=2&seed=4")
    assert resp.status_code == 200
    out = open_response(resp.content)

    assert set(out.data_vars) == {"rr"}
    rr = out["rr"]
    assert rr.dims == ("ensemble_member", "time", "y", "x")
    assert rr.shape == (2, 3, SIZE, SIZE)
    assert rr.attrs["standard_name"] == "rainfall_rate"
    assert rr.attrs["units"] == "mm h-1"
    assert rr.attrs["grid_mapping"] == "crs"
    assert out["crs"].attrs["grid_mapping_name"] == "lambert_azimuthal_equal_area"
    reference = ds.time.values[-1]
    assert out.forecast_reference_time.values == reference
    np.testing.assert_array_equal(out.lead_time.values, np.arange(1, 4) * np.timedelta64(5, "m"))
    np.testing.assert_array_equal(out.time.values, reference + out.lead_time.values)
    np.testing.assert_array_equal(out.x.values, ds.x.values)
    assert out.x.attrs["standard_name"] == "projection_x_coordinate"
    assert out.attrs["Conventions"] == "CF-1.8"
    assert out.attrs["model_sha256"] == loaded.manifest["model_sha256"]
    assert out.attrs["model_name"] == "irene-tiny"
    assert out.attrs["builder_version"] == __version__
    assert out.attrs["seed"] == 4
    assert "comment" not in out.attrs


def test_cf_format_finds_the_variable_by_standard_name(client: TestClient, tmp_path: Path) -> None:
    ds = build_input(nt=6).rename({"rr": "precip"})
    out = open_response(post(client, netcdf4_bytes(ds, tmp_path), "?format=cf&ensemble_size=1").content)
    assert "precip" in out.data_vars


def test_cf_format_converts_units(client: TestClient, tmp_path: Path) -> None:
    ds = build_input(nt=6)
    flux = ds.copy()
    flux["rr"] = ds.rr / 3600
    flux["rr"].attrs = {**ds.rr.attrs, "standard_name": "rainfall_flux", "units": "kg m-2 s-1"}
    a = open_response(post(client, netcdf4_bytes(ds, tmp_path), "?format=cf&seed=1&ensemble_size=2").content)
    b = open_response(post(client, netcdf4_bytes(flux, tmp_path), "?format=cf&seed=1&ensemble_size=2").content)
    np.testing.assert_allclose(a.rr.values, b.rr.values, rtol=1e-4, atol=1e-4)


def test_cf_format_checks_the_time_spacing(client: TestClient, tmp_path: Path) -> None:
    ds = build_input(nt=6)
    ds = ds.assign_coords(time=ds.time.values[0] + np.arange(6) * np.timedelta64(10, "m"))
    resp = post(client, netcdf4_bytes(ds, tmp_path), "?format=cf")
    assert resp.status_code == 422
    assert "5 minutes" in resp.json()["detail"] or "0 days 00:05:00" in resp.json()["detail"]


def test_cf_format_from_a_bare_irene_input_synthesises_times(client: TestClient) -> None:
    out = open_response(post(client, irene_input(), "?format=cf&ensemble_size=1&forecast_steps=2").content)
    assert out["RR"].dims == ("ensemble_member", "time", "y", "x")
    assert out["RR"].attrs["standard_name"] == "rainfall_rate"
    assert "no datetime time coordinate" in out.attrs["comment"]
    np.testing.assert_array_equal(out.y.values, np.arange(SIZE))


def test_raw_body_uploads(client: TestClient) -> None:
    content = irene_input()
    multipart = post(client, content, "?seed=2&ensemble_size=2")
    expected = open_response(multipart.content)["precipitation_forecast"].values
    for method in ("POST", "PUT"):
        for content_type in ("application/x-netcdf", "application/octet-stream"):
            resp = client.request(
                method, "/predict?seed=2&ensemble_size=2", content=content, headers={"Content-Type": content_type}
            )
            assert resp.status_code == 200, (method, content_type, resp.text)
            np.testing.assert_array_equal(open_response(resp.content)["precipitation_forecast"].values, expected)
    resp = client.put("/predict", content=content, headers={"Content-Type": "text/plain"})
    assert resp.status_code == 415


def test_put_accepts_multipart_too(client: TestClient) -> None:
    resp = client.put("/predict", files={"file": ("input.nc", io.BytesIO(irene_input()), "application/x-netcdf")})
    assert resp.status_code == 200


# --- products, dtype -------------------------------------------------------------------


def test_products_match_the_members(client: TestClient, tmp_path: Path) -> None:
    content = netcdf4_bytes(build_input(nt=6), tmp_path)
    resp = post(
        client,
        content,
        "?format=cf&seed=7&products=members,mean,quantiles,exceedance&quantiles=0.1,0.5,0.9&thresholds=0.5,1,5",
    )
    assert resp.status_code == 200
    out = open_response(resp.content)
    members = out.rr.values
    np.testing.assert_allclose(out.rr_mean.values, members.mean(axis=0), rtol=1e-5, atol=1e-6)
    assert out.rr_mean.dims == ("time", "y", "x")
    assert out.rr_mean.attrs["cell_methods"] == "ensemble_member: mean"
    np.testing.assert_array_equal(out["quantile"].values, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(out.rr_quantile.values, np.quantile(members, [0.1, 0.5, 0.9], axis=0), rtol=1e-5)
    assert out.rr_quantile.dims == ("quantile", "time", "y", "x")
    np.testing.assert_array_equal(out.threshold.values, [0.5, 1, 5])
    expected = np.stack([(members > t).mean(axis=0) for t in (0.5, 1, 5)])
    np.testing.assert_allclose(out.rr_exceedance_probability.values, expected, atol=1e-6)
    assert out.rr_exceedance_probability.attrs["units"] == "1"
    assert out.threshold.attrs["units"] == "mm h-1"


def test_products_without_members_drop_them(client: TestClient, tmp_path: Path) -> None:
    content = netcdf4_bytes(build_input(nt=6), tmp_path)
    out = open_response(post(client, content, "?format=cf&products=mean").content)
    assert set(out.data_vars) == {"rr_mean"}
    assert "ensemble_member" not in out.dims


@pytest.mark.parametrize(
    "params",
    [
        "?products=mean",  # format=irene
        "?format=cf&products=median",
        "?format=cf&products=quantiles&quantiles=0.5,1.5",
        "?format=cf&products=exceedance&thresholds=a,b",
        "?format=cf&dtype=float64",
    ],
)
def test_bad_product_parameters_are_422(client: TestClient, params: str) -> None:
    assert post(client, irene_input(), params).status_code == 422


def test_float16_is_packed_int16_and_readable(client: TestClient) -> None:
    content = irene_input()
    full = open_response(post(client, content, "?seed=1&ensemble_size=2").content)["precipitation_forecast"]
    resp = post(client, content, "?seed=1&ensemble_size=2&dtype=float16&complevel=9")
    packed = open_response(resp.content)["precipitation_forecast"]
    raw = open_response(resp.content, mask_and_scale=False)["precipitation_forecast"]
    assert raw.dtype == np.int16
    assert "scale_factor" in raw.attrs
    step = float(full.max() - full.min()) / 65533
    np.testing.assert_allclose(packed.values, full.values, atol=step)


# --- reproducibility and headers -------------------------------------------------------


def test_seed_reproducibility_and_headers(client: TestClient, loaded: LoadedModel) -> None:
    content = irene_input()
    first = post(client, content, "?seed=11&ensemble_size=3", headers={"X-Request-Id": "abc-123"})
    second = post(client, content, "?seed=11&ensemble_size=3")
    other = post(client, content, "?seed=12&ensemble_size=3")
    a, b, c = (open_response(r.content)["precipitation_forecast"].values for r in (first, second, other))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)

    headers = first.headers
    assert headers["X-Request-Id"] == "abc-123"
    assert headers["X-Model-Sha256"] == loaded.manifest["model_sha256"]
    assert headers["X-Model-Name"] == "irene-tiny"
    assert headers["X-Model-Version"] == "0.1"
    assert headers["X-Mlcast-Version"] == mlcast_version()
    assert headers["X-Builder-Version"] == __version__
    assert len(second.headers["X-Request-Id"]) == 32  # generated

    attrs = open_response(first.content).attrs
    assert attrs["request_id"] == "abc-123"
    assert attrs["model_sha256"] == loaded.manifest["model_sha256"]
    assert attrs["seed"] == 11
    assert attrs["model_version"] == "0.1"


def test_request_id_is_on_every_response(client: TestClient) -> None:
    assert client.get("/healthz", headers={"X-Request-Id": "probe-1"}).headers["X-Request-Id"] == "probe-1"
    assert client.get("/healthz", headers={"X-Request-Id": "bad id with spaces"}).headers["X-Request-Id"] != (
        "bad id with spaces"
    )


# --- concurrency -----------------------------------------------------------------------


@pytest.fixture
def slow_forecast(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Make every forecast but the warm-up wait until the returned event is set."""
    release = threading.Event()
    real = serve_module.forecast

    def _slow(*args: Any, **kwargs: Any):
        if threading.current_thread().name != "warm-up":
            assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve_module, "forecast", _slow)
    return release


def test_queue_full_is_429_and_health_stays_responsive(
    make_client: Callable[..., TestClient], slow_forecast: threading.Event
) -> None:
    client = make_client(max_queue=1)
    state = client.app.state.server
    content = irene_input()
    results: list[int] = []

    def request() -> None:
        results.append(post(client, content, "?ensemble_size=1&forecast_steps=1").status_code)

    threads = [threading.Thread(target=request) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        wait_until(lambda: state.gate.admitted == 2 and state.gate.waiting == 1)

        # One running (blocked in the forecast), one waiting: the next is refused
        resp = post(client, content)
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1

        # The health endpoints answer while the inference is in progress
        tic = time.perf_counter()
        for path in ("/healthz", "/health", "/readyz", "/metrics"):
            assert client.get(path).status_code == 200
        assert time.perf_counter() - tic < 5
        metrics = client.get("/metrics").text
        assert "mlcast_icb_queue_depth 1.0" in metrics
        assert "mlcast_icb_predictions_in_progress 2.0" in metrics
    finally:
        slow_forecast.set()
        for thread in threads:
            thread.join(60)
    assert results == [200, 200]
    assert state.gate.admitted == 0


def test_waiting_longer_than_the_timeout_is_503(make_client: Callable[..., TestClient]) -> None:
    client = make_client(request_timeout=0.2)
    with client.app.state.server.gate.model():  # the model is busy
        resp = post(client, irene_input())
    assert resp.status_code == 503
    assert "Retry-After" in resp.headers
    assert post(client, irene_input(), "?ensemble_size=1").status_code == 200


# --- operations ------------------------------------------------------------------------


def test_readyz_waits_for_the_warm_up(make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch) -> None:
    proceed = threading.Event()
    real = serve_module._warm_up

    def _gated(state):
        assert proceed.wait(30)
        real(state)

    monkeypatch.setattr(serve_module, "_warm_up", _gated)
    client = make_client(wait_ready=False)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "warming up"
    assert client.get("/healthz").status_code == 200
    assert client.get("/health").json()["model_loaded"] is True
    assert "mlcast_icb_ready 0.0" in client.get("/metrics").text

    proceed.set()
    wait_until(lambda: client.get("/readyz").status_code == 200)
    assert client.get("/readyz").json() == {"status": "ready"}
    assert "mlcast_icb_ready 1.0" in client.get("/metrics").text


def test_failed_warm_up_keeps_readyz_503(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(*args: Any, **kwargs: Any):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(serve_module, "forecast", _broken)
    client = make_client(wait_ready=False)
    state = client.app.state.server
    wait_until(lambda: state.warm_up_error is not None)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "warm-up failed", "detail": "RuntimeError: CUDA out of memory"}


def test_metrics(make_client: Callable[..., TestClient]) -> None:
    client = make_client()
    assert post(client, irene_input(), "?ensemble_size=1").status_code == 200
    assert post(client, irene_input(), "?forecast_steps=0").status_code == 422
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert 'mlcast_icb_requests_total{endpoint="/predict",method="POST",status="200"} 1.0' in text
    assert 'mlcast_icb_requests_total{endpoint="/predict",method="POST",status="422"} 1.0' in text
    assert 'mlcast_icb_request_duration_seconds_count{endpoint="/predict",method="POST"} 2.0' in text
    # the warm-up and the one prediction
    assert "mlcast_icb_inference_duration_seconds_count 1.0" in text
    assert "mlcast_icb_queue_depth 0.0" in text
    assert 'mlcast_icb_model_info{builder_version="' in text
    assert 'name="irene-tiny"' in text


def test_openapi_documents_the_predict_body(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    predict = spec["paths"]["/predict"]
    assert {"post", "put"} <= set(predict)
    content = predict["post"]["requestBody"]["content"]
    assert {"multipart/form-data", "application/x-netcdf", "application/octet-stream"} <= set(content)
    params = {p["name"] for p in predict["post"]["parameters"]}
    assert {"variable", "forecast_steps", "ensemble_size", "format", "products", "seed", "mask", "dtype"} <= params


# --- configuration and exit codes ------------------------------------------------------


def test_defaults() -> None:
    config = ServeConfig.from_env({})
    assert config == ServeConfig()
    assert (config.host, config.port, config.max_upload_mb, config.max_queue) == ("0.0.0.0", 8000, 100, 2)
    assert config.request_timeout is None
    assert config.model_dir == "/model"


def test_values_are_parsed() -> None:
    env = {
        "MODEL_DIR": "/m",
        "DEVICE": "CPU",
        "HOST": "127.0.0.1",
        "PORT": "9000",
        "MAX_UPLOAD_MB": "5",
        "MAX_QUEUE": "0",
        "REQUEST_TIMEOUT": "2.5",
        "MAX_FORECAST_STEPS": "24",
        "MAX_ENSEMBLE_SIZE": "50",
        "LOG_LEVEL": "debug",
    }
    config = ServeConfig.from_env(env)
    assert config == ServeConfig("/m", "cpu", "127.0.0.1", 9000, 5, 0, 2.5, 24, 50, "DEBUG")
    assert list(describe_config(config)) == list(SERVE_VARIABLES)


def test_every_problem_is_collected() -> None:
    env = {"PORT": "70000", "MAX_QUEUE": "-1", "REQUEST_TIMEOUT": "soon", "DEVICE": "tpu", "MAX_UPLOAD_MB": "0"}
    with pytest.raises(ConfigError) as info:
        ServeConfig.from_env(env)
    message = str(info.value)
    assert "5 problems" in message
    for fragment in ("PORT", "MAX_QUEUE", "REQUEST_TIMEOUT", "DEVICE", "MAX_UPLOAD_MB"):
        assert fragment in message


def test_every_serve_variable_is_known() -> None:
    assert set(SERVE_VARIABLES) == KNOWN_VARIABLES


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace uvicorn's server with a recorder, so ``main`` never binds a port."""
    started: list[Any] = []

    class _Server:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.started = False

        def run(self) -> None:
            started.append(self.config)
            self.started = True

    monkeypatch.setattr(serve_module.uvicorn, "Server", _Server)
    return started


def test_config_error_exits_2_before_binding(no_server: list[Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert main({"PORT": "x", "MAX_QUEUE": "-1", "MAX_QUEU": "3"}) == 2
    assert no_server == []
    err = capsys.readouterr().err
    assert f"Starting mlcast-inference-container-builder {__version__} (serve)" in err
    assert "Did you mean MAX_QUEUE?" in err
    assert "PORT must be an integer" in err and "MAX_QUEUE must be at least 0" in err


def test_missing_bundle_exits_6_before_binding(no_server: list[Any], tmp_path: Path) -> None:
    assert main({"MODEL_DIR": str(tmp_path / "nothing"), "DEVICE": "cpu"}) == 6
    assert no_server == []


def test_cuda_without_a_gpu_exits_4(no_server: list[Any], bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert main({"MODEL_DIR": str(bundle_dir), "DEVICE": "cuda"}) == 4
    assert no_server == []


def test_main_loads_the_model_then_serves(
    no_server: list[Any], bundle_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main({"MODEL_DIR": str(bundle_dir), "DEVICE": "cpu", "HOST": "127.0.0.1", "PORT": "8123"}) == 0
    (uvicorn_config,) = no_server
    assert (uvicorn_config.host, uvicorn_config.port) == ("127.0.0.1", 8123)
    err = capsys.readouterr().err
    assert "Effective configuration" in err
    assert "irene-tiny" in err
    assert "req=-" in err


def test_lifespan_loads_the_model_when_not_given(config: ServeConfig) -> None:
    with TestClient(create_app(config)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
