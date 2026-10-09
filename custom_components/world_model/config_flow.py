"""Config flow for the local world model bridge."""

import logging

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_URL

from .const import CONF_BRIDGE_TOKEN, DOMAIN

_LOGGER = logging.getLogger(__name__)


class WorldModelConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            token = user_input[CONF_BRIDGE_TOKEN]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url + "/api/ha/state",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status != 200:
                            errors["base"] = "cannot_connect"
                        else:
                            await response.json()
            except (aiohttp.ClientError, TimeoutError, ValueError) as error:
                _LOGGER.debug("World model bridge validation failed: %s", error)
                errors["base"] = "cannot_connect"
            if not errors:
                await self.async_set_unique_id(url)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Home World Model",
                    data={CONF_URL: url, CONF_BRIDGE_TOKEN: token},
                )

        schema = vol.Schema({
            vol.Required(CONF_URL, default="http://home-world-model.local:8765"): str,
            vol.Required(CONF_BRIDGE_TOKEN): str,
        })
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)