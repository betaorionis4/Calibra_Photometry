"""
Star names -> coordinates (v4.30).

Order: 1. local GCVS list (if a local catalogue folder with a GCVS list is set and
'Use local copies' is on - the default), 2. SIMBAD/Sesame online (astropy
SkyCoord.from_name), only if going online is allowed. The callers keep their
own last fallback (AAVSO VSX / VSP via python.calibration.resolve_auid).
"""
from python import local_catalogs as _lc

LAST_SOURCE = {'name': '', 'source': ''}


def from_name(name):
    """Drop-in for astropy SkyCoord.from_name: local GCVS list first, then SIMBAD.
    Returns a SkyCoord (ICRS); raises an exception if the name is not found."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = _lc.gcvs_lookup(name)
    if t is not None:
        LAST_SOURCE.update(name=name, source='local GCVS list')
        return SkyCoord(ra=float(t['ra']) * u.deg, dec=float(t['dec']) * u.deg, frame='icrs')
    if not _lc.online_allowed():
        raise ValueError(f"'{name}' is not in the local GCVS list and going online is switched off.")
    from python import online as _online
    if not _online.available('simbad'):
        raise ValueError(f"'{name}' is not in the local GCVS list and SIMBAD was not reachable a short while ago.")
    _online.set_timeout()
    try:
        c = SkyCoord.from_name(name)
    except Exception as e:
        # unknown names raise NameResolveError ('Unable to find coordinates'); only network problems count
        if 'unable to find' not in str(e).lower():
            _online.failed('simbad', e)
        raise
    _online.ok('simbad')
    LAST_SOURCE.update(name=name, source='SIMBAD')
    return c
