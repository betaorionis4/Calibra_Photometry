"""
Availability of online services (v4.34).

When a server (VizieR, SIMBAD, AAVSO) does not answer, every further request would
wait for the full timeout again. After a failure the service is therefore skipped for
PAUSE seconds; local data and caches are used meanwhile. A message is printed once.
"""
import threading
import time

PAUSE = 600.0          # s a failed service is skipped
TIMEOUT = 20           # s, astropy/astroquery remote timeout (was 60)

_DOWN = {}
_LOCK = threading.Lock()
NAMES = {'vizier': 'VizieR (CDS)', 'simbad': 'SIMBAD (CDS)', 'aavso': 'AAVSO'}


def available(service):
    with _LOCK:
        t = _DOWN.get(service)
        if t and time.time() - t[0] > PAUSE:
            _DOWN.pop(service, None)
            t = None
    return t is None


def failed(service, err):
    """Record a failed request (network error, timeout, server error)."""
    with _LOCK:
        first = service not in _DOWN
        _DOWN[service] = (time.time(), str(err))
    if first:
        print(f"{NAMES.get(service, service)} not reachable ({str(err)[:120]}) - online requests to it are "
              f"skipped for {PAUSE / 60:.0f} min; local data and caches are used.")


def ok(service):
    with _LOCK:
        _DOWN.pop(service, None)


def status():
    """{service: (since, message)} of the services currently skipped."""
    with _LOCK:
        return {k: v for k, v in _DOWN.items() if time.time() - v[0] <= PAUSE}


def set_timeout():
    """Shorter remote timeout for astropy/astroquery requests."""
    try:
        from astropy.utils.data import Conf
        Conf.remote_timeout.set(TIMEOUT)
    except Exception:
        pass
