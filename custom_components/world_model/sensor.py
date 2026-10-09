"""Read-only model-state sensors."""

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN


SENSORS = (
    ("status", "Model status", "model_status"),
    ("activity", "Current activity", "current_activity"),
    ("next_action", "Predicted next action", "next_action"),
    ("action_score", "User action score", "action_score"),
    ("event_count", "Events received", "events_received"),
)


async def async_setup_entry(hass, entry, async_add_entities):
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(WorldModelSensor(coordinator, key, name, field)
                       for key, name, field in SENSORS)


class WorldModelSensor(CoordinatorEntity, SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator, key, name, field):
        super().__init__(coordinator)
        self._field = field
        self._attr_name = name
        self._attr_unique_id = f"{DOMAIN}_{key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, "local_world_model")},
            "name": "Home World Model",
            "manufacturer": "Local server",
            "model": "Temporal action predictor",
        }
        if key == "action_score":
            self._attr_native_unit_of_measurement = PERCENTAGE
        if key == "event_count":
            self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self):
        data = self.coordinator.data or {}
        prediction = data.get("prediction") or {}
        if self._field == "next_action":
            candidates = prediction.get("candidates") or []
            if not candidates or data.get("warmup_minutes", 0) > 0:
                return None
            return candidates[0].get("action")
        if self._field == "action_score":
            score = prediction.get("action_score")
            return round(score * 100, 1) if score is not None else None
        value = data.get(self._field)
        if self._field == "model_status" and value == "ready" and data.get("warmup_minutes", 0) > 0:
            return "warming"
        return value

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data or {}
        prediction = data.get("prediction") or {}
        if self._field in ("next_action", "action_score"):
            return {
                "candidates": prediction.get("candidates", []),
                "other_action_score": prediction.get("other_action_score"),
                "scores_are_uncalibrated": True,
                "model_device": data.get("model_device"),
                "context_warmup_minutes": data.get("warmup_minutes"),
            }
        if self._field == "status":
            return {
                "ha_connected": data.get("connected"),
                "connection_state": data.get("connection_state"),
                "events_received": data.get("events_received"),
                "last_event_at": data.get("last_event_at"),
                "model_device": data.get("model_device"),
                "training_samples": data.get("model_samples"),
                "model_features": data.get("model_features"),
            }
        return None