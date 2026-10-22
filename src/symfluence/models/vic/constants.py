"""Write worker-local VIC image-driver constants without losing other settings."""
from pathlib import Path
import math

LEGACY_KEYS = {'MIN_RAIN_TEMP': 'SNOW_MIN_RAIN_TEMP',
               'MAX_SNOW_TEMP': 'SNOW_MAX_SNOW_TEMP'}


def target_snow_constants(content, source_dir, settings_dir, values):
    """Return a global file using a private constants file for explicit overrides.

    Unsupported legacy global keys require explicit replacement, avoiding silent
    activation of thresholds that VIC previously ignored.
    """
    lines = content.splitlines()
    paths = [line.split()[1] for line in lines
             if line.split() and line.split()[0].upper() == 'CONSTANTS']
    if len(paths) > 1:
        raise ValueError('Multiple VIC CONSTANTS files are ambiguous')
    constants = ''
    if paths:
        source = Path(paths[0])
        if not source.is_absolute():
            source = Path(source_dir) / source
        constants = source.read_text()
    for line in lines:
        parts = line.split()
        if parts and parts[0].upper() in LEGACY_KEYS:
            target = LEGACY_KEYS[parts[0].upper()]
            if target not in values:
                raise ValueError(f'{parts[0]} is ignored by VIC; explicitly supply {target}')
    if not values:
        return content
    effective = {'SNOW_MIN_RAIN_TEMP': -.5, 'SNOW_MAX_SNOW_TEMP': .5}
    for line in constants.splitlines():
        parts = line.split()
        if parts and parts[0].upper() in effective:
            effective[parts[0].upper()] = float(parts[1])
    effective.update(values)
    lo, hi = effective['SNOW_MIN_RAIN_TEMP'], effective['SNOW_MAX_SNOW_TEMP']
    if not all(math.isfinite(v) and -10 <= v <= 10 for v in (lo, hi)) or lo >= hi:
        raise ValueError('VIC rain/snow thresholds require -10 <= minimum < maximum <= 10 C')
    retained = [line for line in constants.splitlines()
                if not line.split() or line.split()[0].upper() not in values]
    retained.extend(f'{key} {value:.17g}' for key, value in values.items())
    destination = Path(settings_dir) / 'vic_calibration_constants.txt'
    destination.write_text('\n'.join(retained) + '\n')
    clean = [line for line in lines if not line.split() or
             line.split()[0].upper() not in {*LEGACY_KEYS, 'CONSTANTS'}]
    clean.append(f'CONSTANTS {destination.resolve()}')
    return '\n'.join(clean) + '\n'
