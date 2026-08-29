"""The deployment files, read as data.

Compose and Caddy configuration is not code, and it is the part of T24 that fails
silently: a published port that should not be, a proxy-hop count that stops matching
the number of proxies, a service that does not come back after the box reboots. None
of those raise anything. They are just wrong, on a public URL.

So the properties that matter are asserted here rather than checked by eye. Each one
below is the safe half of a pair — the other half is in `paddock.api.ratelimit` or in
the iptables rules on the box — and this file exists because the two halves are
edited months apart.
"""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "docker-compose.yml"
DEPLOY = ROOT / "docker-compose.deploy.yml"
CADDYFILE = ROOT / "deploy" / "Caddyfile"
LOCK = ROOT / "uv.lock"

PUBLIC_PORTS = {"80", "443"}


def _compose(path: pathlib.Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text())
    return loaded


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return _compose(BASE)


@pytest.fixture(scope="module")
def deploy() -> dict[str, Any]:
    return _compose(DEPLOY)


@pytest.fixture(scope="module")
def caddyfile() -> str:
    return CADDYFILE.read_text()


def _published(service: dict[str, Any]) -> list[str]:
    return [str(entry) for entry in service.get("ports", [])]


# ── Nothing but Caddy is reachable from outside the box ─────────────────────────


def test_the_base_stack_publishes_only_to_loopback(base: dict[str, Any]) -> None:
    """A published port with no address binds every interface, and the box has a
    public IP. Postgres on the internet is the whole corpus, readable."""
    for name, service in base["services"].items():
        for entry in _published(service):
            assert entry.startswith("127.0.0.1:"), f"{name} publishes {entry} on all interfaces"


def test_only_caddy_takes_traffic_from_the_internet(deploy: dict[str, Any]) -> None:
    for name, service in deploy["services"].items():
        published = {entry.split(":")[-2] for entry in _published(service) if ":" in entry}
        if name == "caddy":
            continue
        assert not (published & PUBLIC_PORTS), f"{name} publishes a public port"


def test_caddy_serves_both_http_and_https(deploy: dict[str, Any]) -> None:
    """80 as well as 443: Let's Encrypt answers its HTTP-01 challenge there, and
    without it Caddy cannot get a certificate at all."""
    published = {entry.split(":")[-2] for entry in _published(deploy["services"]["caddy"])}

    assert published >= PUBLIC_PORTS


# ── The forwarded address is trusted by exactly as many hops as exist ───────────


def test_the_deployed_api_trusts_exactly_one_proxy(deploy: dict[str, Any]) -> None:
    """`api_trusted_proxy_hops` decides which `X-Forwarded-For` entry the rate
    limiter meters. Set higher than the number of proxies actually in front, it
    reads an entry the caller wrote, and one header per request buys a fresh
    budget. Set lower, every visitor is metered as Caddy and one of them starves
    the rest. The stack runs one Caddy, so the number is 1."""
    environment = deploy["services"]["api"]["environment"]

    assert environment["API_TRUSTED_PROXY_HOPS"] == 1


def test_the_local_stack_trusts_no_proxy(base: dict[str, Any]) -> None:
    """Nothing stands in front of it, so the header is whatever a caller sent."""
    environment = base["services"]["api"].get("environment", {})

    assert int(environment.get("API_TRUSTED_PROXY_HOPS", 0)) == 0


# ── The stack comes back by itself ──────────────────────────────────────────────


def test_every_deployed_service_restarts_after_a_reboot(deploy: dict[str, Any]) -> None:
    """The box is an Always Free instance. Oracle reboots it for maintenance, and
    nobody is watching when it happens."""
    base_services = _compose(BASE)["services"]
    names = set(base_services) | set(deploy["services"])
    for name in sorted(names):
        # Compose merges the two files per service, so a key set in the base file
        # and left alone in the overlay still applies.
        merged = {**base_services.get(name, {}), **deploy["services"].get(name, {})}
        assert merged.get("restart") == "unless-stopped", f"{name} does not restart"


# ── Caddy sends each path to the process that owns it ───────────────────────────


@pytest.mark.parametrize("path", ["/ask", "/coverage", "/health"])
def test_the_api_paths_reach_the_api(caddyfile: str, path: str) -> None:
    """Spec §3: the API is the product and the demo is one client of it. A visitor
    who wants to curl `/ask` should be able to."""
    assert path in caddyfile


def test_the_domain_is_not_hardcoded(caddyfile: str) -> None:
    """It comes from the environment, so the file in the repository names no host
    and a second deployment needs no edit."""
    assert "paddock-hk.duckdns.org" not in caddyfile
    assert "PADDOCK_DOMAIN" in caddyfile


def test_caddy_reaches_the_other_two_by_service_name(caddyfile: str) -> None:
    """Over the compose network, not through a published port."""
    assert "api:8000" in caddyfile
    assert "ui:8501" in caddyfile


# ── The image carries nothing the box cannot use ────────────────────────────────


def test_the_locked_torch_pulls_no_gpu_libraries() -> None:
    """torch's Linux dependencies are marked `sys_platform == 'linux'` with no
    architecture, so the default PyPI wheel drags the whole CUDA stack onto a box
    with no GPU — gigabytes of it, and `nvidia-nccl-cu13` even ships an aarch64
    build, so nothing fails loudly. `pyproject.toml` points torch at PyTorch's CPU
    index on Linux instead.

    Asserted here because the failure is invisible: the next `uv lock` can bring it
    all back and the only symptom is a slower build and a fuller disk.
    """
    lock = tomllib.loads(LOCK.read_text())
    torch = next(package for package in lock["package"] if package["name"] == "torch")

    gpu_only = sorted(
        dependency["name"]
        for dependency in torch.get("dependencies", [])
        if dependency["name"].startswith(("nvidia-", "cuda-")) or dependency["name"] == "triton"
    )

    assert gpu_only == []
