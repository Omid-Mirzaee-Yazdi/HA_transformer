"""Local dashboard and read-only Home Assistant reporting API."""

import asyncio
import ipaddress
import logging
import ssl
from contextlib import asynccontextmanager
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ha_world_model.runtime import LiveRuntime


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
runtime = LiveRuntime()
logger = logging.getLogger("ha_world_model")


class ServerSettings(BaseModel):
    ha_url: str = Field(min_length=8, max_length=512)
    access_token: str = Field(min_length=20, max_length=2048)
    timezone: str = Field(min_length=1, max_length=128)
    ca_cert_path: str = Field(default="", max_length=4096)


def is_loopback(request):
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except (AttributeError, ValueError):
        return False


def require_loopback(request):
    if not is_loopback(request):
        raise HTTPException(status_code=403, detail="This endpoint is available from the Mac only")


@asynccontextmanager
async def lifespan(_app):
    await runtime.start()
    yield
    await runtime.stop()


app = FastAPI(title="Home World Model", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; form-action 'self'; frame-ancestors 'none'")
    return response


@app.get("/")
async def dashboard(request: Request):
    require_loopback(request)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{name}")
async def static_asset(name: str, request: Request):
    require_loopback(request)
    if name not in {"app.js", "app.css"}:
        raise HTTPException(status_code=404)
    return FileResponse(STATIC_DIR / name)


@app.get("/api/dashboard")
async def dashboard_state(request: Request):
    require_loopback(request)
    return runtime.dashboard_state()


@app.post("/api/config")
async def update_config(settings: ServerSettings, request: Request):
    require_loopback(request)
    if not settings.ha_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Home Assistant URL must use http:// or https://")
    try:
        ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError as error:
        raise HTTPException(status_code=422, detail="Unknown IANA timezone") from error
    ca_cert_path = settings.ca_cert_path.strip()
    if ca_cert_path:
        ca_file = Path(ca_cert_path).expanduser()
        if not ca_file.is_file():
            raise HTTPException(status_code=422, detail="CA certificate path is not a file")
        try:
            ssl.create_default_context().load_verify_locations(cafile=str(ca_file))
        except (OSError, ssl.SSLError) as error:
            raise HTTPException(status_code=422, detail="CA certificate file could not be loaded") from error
        ca_cert_path = str(ca_file.resolve())
    runtime.update_config(settings.ha_url, settings.access_token, settings.timezone, ca_cert_path)
    return {"saved": True}


@app.post("/api/retrain")
async def retrain(request: Request):
    require_loopback(request)
    if runtime._training_task and not runtime._training_task.done():
        raise HTTPException(status_code=409, detail="Model training is already running")
    runtime._training_task = asyncio.create_task(runtime.retrain())
    return {"started": True}


@app.get("/api/ha/state")
async def home_assistant_sensor_state(request: Request):
    expected = runtime.config.get("bridge_token", "")
    provided = request.headers.get("authorization", "")
    if not expected or not provided.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    import secrets
    if not secrets.compare_digest(provided[7:], expected):
        raise HTTPException(status_code=403, detail="Invalid bridge token")
    return runtime.ha_sensor_state()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8765, access_log=False)


if __name__ == "__main__":
    main()