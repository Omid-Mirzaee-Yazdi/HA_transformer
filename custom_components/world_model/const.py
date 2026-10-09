"""Constants for the Home World Model integration."""

from datetime import timedelta

DOMAIN = "world_model"
CONF_BRIDGE_TOKEN = "bridge_token"
PLATFORMS = ("sensor",)
SCAN_INTERVAL = timedelta(seconds=10)