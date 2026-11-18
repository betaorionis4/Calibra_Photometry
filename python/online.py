"""
Availability of online services (v4.34).

When a server (VizieR, SIMBAD, AAVSO) does not answer, every further request would
wait for the full timeout again. After a failure the service is therefore skipped for
PAUSE seconds; local data and caches are used meanwhile. A message is printed once.
v4.46: the pause doubles with every further failure in a row (10 -> 20 -> 40 -> 60 min max),
so a server that is down for days costs one 20 s timeout per hour at most.
"""
import threading
import time

PAUSE = 600.0          # s a failed service is skipped (first failure)
PAUSE_MAX = 3600.0     # s, longest pause after repeated failures (v4.46)
TIMEOUT = 20           # s, astropy/astroquery remote timeout (was 60)

_DOWN = {}
_FAILS = {}            # consecutive failures per service (v4.46)
_LOCK = threading.Lock()
NAMES = {'vizier': 'VizieR (CDS)', 'simbad': 'SIMBAD (CDS)', 'aavso': 'AAVSO'}


def pause(service):
    """Current pause (s) for a service after its consecutive failures."""
    n = max(1, _FAILS.get(service, 1))
    return min(PAUSE * 2 ** (n - 1), PAUSE_MAX)


def available(service):
    with _LOCK:
        t = _DOWN.get(service)
        if t and time.time() - t[0] > pause(service):
            _DOWN.pop(service, None)
            t = None
    return t is None


def failed(service, err):
    """Record a failed request (network error, timeout, server error)."""
    with _LOCK:
        first = service not in _DOWN
        _DOWN[service] = (time.time(), str(err))
        _FAILS[service] = _FAILS.get(service, 0) + 1
        p_ = pause(service)
    if first:
        print(f"{NAMES.get(service, service)} not reachable ({str(err)[:120]}) - online requests to it are "
              f"skipped for {p_ / 60:.0f} min; local data and caches are used.")


def ok(service):
    with _LOCK:
        _DOWN.pop(service, None)
        _FAILS.pop(service, None)


def status():
    """{service: (since, message)} of the services currently skipped."""
    with _LOCK:
        return {k: v for k, v in _DOWN.items() if time.time() - v[0] <= pause(k)}


def set_timeout():
    """Shorter remote timeout for astropy/astroquery requests."""
    try:
        from astropy.utils.data import Conf
        Conf.remote_timeout.set(TIMEOUT)
    except Exception:
        pass
