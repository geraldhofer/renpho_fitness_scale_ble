"""Diagnostics support for renpho_fitness_scale_ble.

Wired up via HA's built-in "Download Diagnostics" button on the device
card. Returns a single JSON blob with the config entry + coordinator
state, with PII redacted.

Redacted fields:
  - ``address``                — BLE MAC of the scale (privacy). Only its
                                 OUI is reported (``address_oui``): models
                                 are told apart by it. Any MAC-like pattern
                                 elsewhere in the dump (the entry title
                                 embeds the scale's, scanner names embed the
                                 proxy's / adapter's) keeps its OUI and
                                 loses its device-specific half.
  - ``person_entity``          — links to a HA person entity (privacy)
  - ``mobile_notify_services`` — companion-app service names usually contain
                                 the user's phone hostname
  - ``name``                   — the user's display name
  - ``birthdate`` / ``height`` / ``sex`` — user PII
  - ``user_id``                — a slug *derived from the display name*
                                 ("Alice" → ``alice``), so it is replaced by
                                 a positional alias (``user_1``, ``user_2``…)
                                 that is consistent across the whole dump

What we keep verbatim:
  - ``measurement_id`` strings (opaque uuid hex)
  - Weight values, body fat, resistance — useful for debugging, not PII
    in the same way an address is. The same goes for the frames in the
    library's session trace (``coordinator.scale.trace``): they carry the
    measurements, while the personal bytes of outgoing profile frames are
    masked by the library (``mask_profiles=True``).
"""

from __future__ import annotations

import re
import time
from typing import Any

from homeassistant.components import bluetooth
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from .const import (
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PENDING_STATE,
    CONF_PERSON_ENTITY,
    CONF_ROUTER_STATE,
    CONF_SEX,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    DOMAIN,
)
from .coordinator import ScaleDataUpdateCoordinator

# Top-level entry.data keys to redact wholesale (the router_state contains
# every persisted weight measurement; not strictly PII but huge and noisy
# for a diagnostics dump — we summarize separately below).
_TO_REDACT = {
    "address",
    "unique_id",  # entry.unique_id is the BLE MAC
    CONF_USER_NAME,
    CONF_PERSON_ENTITY,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_SEX,
    CONF_ROUTER_STATE,
    CONF_PENDING_STATE,
}

# A MAC in any of the spellings that reach a dump: colon-separated (titles,
# scanner names), or BlueZ's ``dev_AA_BB_…`` object paths and dashed forms
# (exception text quoted in the library's trace events).
_MAC_RE = re.compile(
    r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{2})([:_-])([0-9A-Fa-f]{2})\2([0-9A-Fa-f]{2})"
    r"(?:\2[0-9A-Fa-f]{2}){3}(?![0-9A-Fa-f])"
)


def _mask_macs(value: Any) -> Any:
    """``value`` with the device-specific half of every MAC masked.

    Strings are masked, containers are walked, anything else is returned
    as-is. The OUI stays: it is what tells scale models (and proxy
    hardware) apart, and it identifies no one.
    """
    if isinstance(value, str):
        return _MAC_RE.sub(
            lambda m: m.group(2).join(
                (m.group(1), m.group(3), m.group(4), *["XX"] * 3)
            ),
            value,
        )
    if isinstance(value, dict):
        return {key: _mask_macs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask_macs(item) for item in value]
    return value


def _address_oui(address: str) -> str | None:
    """First three octets of a MAC address; None when it isn't one (macOS
    hands out CoreBluetooth UUIDs)."""
    octets = address.split(":")
    if len(octets) != 6:
        return None
    return ":".join(octets[:3]).upper()


def _scale_diagnostic_info(scale: Any) -> dict[str, Any] | None:
    """The library's snapshot of the scale, in its profile-masked form.

    None when there is no scale object, or when the installed library
    predates ``get_diagnostic_info``. The unmasked ``diagnostic_info``
    property is deliberately never the fallback: its trace carries the sex,
    age / birth date and height this dump redacts everywhere else.
    """
    get_info = getattr(scale, "get_diagnostic_info", None)
    if get_info is None:
        return None
    try:
        return _mask_macs(get_info(mask_profiles=True))
    except Exception as ex:  # noqa: BLE001 - a dump must never fail on this
        return {"error": _mask_macs(f"{type(ex).__name__}: {ex}")}


def _last_service_info(
    hass: HomeAssistant, address: str
) -> bluetooth.BluetoothServiceInfoBleak | None:
    """HA's most recent advertisement from the scale, on any scanner.

    The connectable history is asked first — it is the path a GATT session
    would use — then the non-connectable one, which also holds what
    passive-only proxies relay and is the only one a broadcast scale is in.
    """
    for connectable in (True, False):
        try:
            info = bluetooth.async_last_service_info(
                hass, address, connectable=connectable
            )
        except Exception:  # noqa: BLE001 - bluetooth not set up
            return None
        if info is not None:
            return info
    return None


def _enum_value(member: Any) -> Any:
    """An enum member's value; anything else (None included) as-is."""
    return getattr(member, "value", member)


def _scanner_path(
    hass: HomeAssistant, info: bluetooth.BluetoothServiceInfoBleak
) -> dict[str, Any]:
    """Through which scanner HA last heard the scale, and how well."""
    source = getattr(info, "source", None)
    scanner = None
    if source:
        try:
            scanner = bluetooth.async_scanner_by_source(hass, source)
        except Exception:  # noqa: BLE001
            scanner = None
    if scanner is None:
        via = "unknown"
    elif isinstance(scanner, bluetooth.BaseHaRemoteScanner):
        via = "remote_proxy"
    else:
        via = "local_adapter"
    scanner_type = getattr(getattr(scanner, "details", None), "scanner_type", None)
    seen = getattr(info, "time", None)
    return {
        "via": via,
        "scanner_type": getattr(scanner_type, "value", None),
        # A passive scanner never relays scan responses, and some QN models
        # carry their model identifier only there — so this is what tells
        # "identifier never reaches HA" from "packet missed on a weak link"
        # when the advertisement shows no manufacturer data.
        "scanner_mode": _enum_value(getattr(scanner, "current_mode", None)),
        "scanner_requested_mode": _enum_value(getattr(scanner, "requested_mode", None)),
        "scanner_name": _mask_macs(getattr(scanner, "name", None)),
        "scanner_source": _mask_macs(source),
        "connectable": getattr(info, "connectable", None),
        "rssi": getattr(info, "rssi", None),
        "seconds_since_seen": (
            round(time.monotonic() - seen, 1)
            if isinstance(seen, (int, float))
            else None
        ),
    }


def _advertisement_from_service_info(
    info: bluetooth.BluetoothServiceInfoBleak, protocol: str
) -> dict[str, Any] | None:
    """The library's ``advertisement`` block, rebuilt from HA's history.

    Covers dumps taken when the library has not heard the scale itself
    (fresh restart, scale asleep), and gives a second, independent view when
    it has. Adds the model code and label, which the library's own snapshot
    reports next to — not inside — this block.
    None if the installed library lacks the masking helper.
    """
    try:
        from renpho_escs20m import mask_mac_echo, model_label

        from .config_flow import advertised_model_code
    except ImportError:
        return None
    seen = getattr(info, "time", None)
    model_code = advertised_model_code(protocol, info.manufacturer_data, info.address)
    return {
        "timestamp": (
            int(time.time() - (time.monotonic() - seen))
            if isinstance(seen, (int, float))
            else None
        ),
        "local_name": _mask_macs(info.name),
        "rssi": info.rssi,
        "service_uuids": list(info.service_uuids or []),
        "manufacturer_data": {
            f"0x{company_id:04x}": mask_mac_echo(bytes(payload), info.address)
            for company_id, payload in info.manufacturer_data.items()
        },
        "model_code": None if model_code is None else f"0x{model_code:04x}",
        "model_label": model_label(protocol, model_code),
    }


async def _integration_version(hass: HomeAssistant) -> str | None:
    try:
        integration = await async_get_integration(hass, DOMAIN)
    except Exception:  # noqa: BLE001
        return None
    return str(integration.version) if integration.version else None


def _library_version() -> str | None:
    try:
        import renpho_escs20m
    except ImportError:
        return None
    return getattr(renpho_escs20m, "__version__", None)


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a Renpho scale config entry."""
    coordinator: ScaleDataUpdateCoordinator | None = hass.data.get(DOMAIN, {}).get(
        entry.entry_id
    )

    # user_id is a slug of the display name, so it is aliased rather than
    # kept: by position, entry data first, so that the entry's profiles,
    # the coordinator summaries and the pending candidates all agree.
    aliases: dict[str, str] = {}

    def alias(user_id: str) -> str:
        return aliases.setdefault(user_id, f"user_{len(aliases) + 1}")

    data = async_redact_data(dict(entry.data), _TO_REDACT)
    profiles = data.get(CONF_USER_PROFILES)
    if isinstance(profiles, list):
        data[CONF_USER_PROFILES] = [
            {**p, CONF_USER_ID: alias(p[CONF_USER_ID])}
            if isinstance(p, dict) and isinstance(p.get(CONF_USER_ID), str)
            else p
            for p in profiles
        ]

    diag: dict[str, Any] = {
        "integration_version": await _integration_version(hass),
        "library_version": _library_version(),
        "entry": {
            "version": entry.version,
            "title": _mask_macs(entry.title),
            "source": entry.source,
            "data": data,
        },
    }

    if coordinator is None:
        diag["coordinator"] = None
        return diag

    # Per-user history summaries (counts only — full history is in
    # router_state which we redact above; counts are what's diagnostic).
    user_history: dict[str, dict[str, Any]] = {}
    for profile in coordinator.get_user_profiles():
        user_id = profile.get("user_id", "")
        history = coordinator.get_recent_user_history(user_id)
        latest = history[-1] if history else None
        user_history[alias(user_id)] = {
            "history_count": len(history),
            "latest_timestamp": (
                latest.timestamp.isoformat() if latest is not None else None
            ),
            "latest_weight_kg": latest.weight_kg if latest is not None else None,
            "latest_has_body_fat": (
                bool((latest.raw or {}).get("body_fat") is not None)
                if latest is not None
                else False
            ),
        }

    # Pending-measurement summary — drop raw_measurement (not JSON-safe)
    # and notified_mobile_services (PII), keep the rest.
    pending_redacted: dict[str, dict[str, Any]] = {}
    for mid, info in coordinator.get_pending_measurements().items():
        pending_redacted[mid] = {
            k: v
            for k, v in info.items()
            if k not in {"raw_measurement", "notified_mobile_services"}
        }
        candidates = pending_redacted[mid].get("candidates")
        if isinstance(candidates, list):
            pending_redacted[mid]["candidates"] = [alias(c) for c in candidates]

    scale = coordinator._scale
    scale_info = _scale_diagnostic_info(scale) if scale is not None else None

    # What HA's own bluetooth stack last heard from the scale. Independent
    # of the library's scanner, so it still answers "is the scale reaching
    # HA at all, and through what" when no session ever started.
    service_info = _last_service_info(hass, coordinator.address)
    # Always reported, even next to the library's own snapshot: the two come
    # from different scanners (the library's is the integration's scanner —
    # a native passive one or a proxy subscription — HA's is whichever of its
    # scanners heard the scale last), so they can disagree, and a field
    # missing from one but not the other says where it is being lost.
    ha_advertisement = None
    if service_info is not None:
        try:
            ha_advertisement = _advertisement_from_service_info(
                service_info, coordinator._protocol
            )
        except Exception as ex:  # noqa: BLE001 - a dump must never fail on this
            ha_advertisement = {"error": _mask_macs(f"{type(ex).__name__}: {ex}")}

    diag["coordinator"] = {
        "device_name": _mask_macs(coordinator.device_name),
        "address_oui": _address_oui(coordinator.address),
        "protocol": coordinator._protocol,
        "display_unit": coordinator.get_display_unit().name,
        "user_count": len(coordinator.get_user_profiles()),
        "user_ids": [
            alias(p.get("user_id", "")) for p in coordinator.get_user_profiles()
        ],
        "user_has_body_metrics": {
            alias(p.get("user_id", "")): bool(p.get("body_metrics_enabled", False))
            for p in coordinator.get_user_profiles()
        },
        "user_history": user_history,
        "pending_count": len(pending_redacted),
        "pending": pending_redacted,
        # A client object exists — NOT a live BLE link: the library connects
        # per weigh-in, and the trace says whether a session is open.
        "scale_client_present": scale is not None,
        "scale_firmware_revision": (
            getattr(scale, "firmware_revision", None) if scale else None
        ),
        "scale_battery_level": (
            getattr(scale, "battery_level", None) if scale else None
        ),
        "scale": scale_info,
        "scanner_path": (
            _scanner_path(hass, service_info) if service_info is not None else None
        ),
        "ha_advertisement": ha_advertisement,
    }
    return diag
