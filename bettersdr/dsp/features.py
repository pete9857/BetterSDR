"""Cheap spectral features for the classifier.

These read a PSD that `dsp/psd.py` already produced rather than re-measuring
the air themselves. That is the same rule the detector follows: if a feature
measured the band differently from the picture on screen, "the app says this
station has HD but I cannot see it" becomes possible.

Every feature here is deliberately a rule with a number attached, not a model.
The product promise is that the app can say *why* it thinks something, so a
feature that cannot be put into one plain sentence does not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- HD Radio (NRSC-5 hybrid IBOC) -----------------------------------------
#
# An HD station keeps its ordinary analog FM carrier and adds OFDM sidebands
# either side of it. Those sidebands are the giveaway: they are *flat*, they
# are symmetric, and they sit in a region where an analog-only station has
# nothing but noise.

# Where the primary digital sidebands live, either side of the analog carrier.
HD_SIDEBAND_LO_HZ = 129_000
HD_SIDEBAND_HI_HZ = 198_000
# Past the sidebands but short of the next allocation: bare noise floor on a
# clean channel, which is what the shoulders are measured against.
HD_REFERENCE_LO_HZ = 240_000
HD_REFERENCE_HI_HZ = 380_000
# The analog signal itself, used only to report how far down the digital part
# sits. Broadcasters run anywhere from -20 to -10 dBc.
HD_ANALOG_CORE_HZ = 60_000

# Each sideband must stand this far above the noise floor beyond 240 kHz.
HD_MIN_SNR_DB = 6.0
# Sub-band levels within one sideband must vary less than this. OFDM is flat;
# the skirt of an over-deviating analog station slopes.
HD_MAX_FLATNESS_DB = 6.0
# IBOC is symmetric. A big imbalance means one shoulder is really a neighbour
# on the adjacent channel, not a digital sideband.
HD_MAX_IMBALANCE_DB = 10.0
# How many pieces each sideband is split into to measure flatness.
_HD_FLATNESS_BANDS = 8


@dataclass(frozen=True)
class HdRadio:
    """Whether an FM station is also broadcasting HD Radio."""

    present: bool
    lower_snr_db: float
    upper_snr_db: float
    level_dbc: float
    flatness_db: float

    @property
    def summary(self) -> str:
        """One sentence for Simple mode: no numbers, no acronyms expanded."""
        if self.present:
            return (
                "This station is also broadcasting in HD, so it may carry extra "
                "channels you cannot hear on a normal radio."
            )
        return "This station is broadcasting the ordinary way only."

    @property
    def detail(self) -> str:
        """The reasoning, with the numbers, for Standard and Expert."""
        if self.present:
            return (
                f"Flat digital sidebands 129-198 kHz either side of the carrier, "
                f"{abs(self.level_dbc):.0f} dB below the analog signal and "
                f"{min(self.lower_snr_db, self.upper_snr_db):.0f} dB above the "
                f"noise floor -> HD Radio"
            )
        return (
            f"No flat sidebands at 129-198 kHz "
            f"({self.lower_snr_db:+.0f} dB lower, {self.upper_snr_db:+.0f} dB upper, "
            f"over the noise floor) -> analog only"
        )


def _offsets(bins: int, bin_width_hz: float) -> np.ndarray:
    """Frequency of each bin of an fftshifted spectrum, relative to centre."""
    return (np.arange(bins) - bins // 2) * bin_width_hz


def _mean_power_db(power: np.ndarray, selection: np.ndarray) -> float:
    """Average in linear power, then convert - averaging dB is not average power."""
    if not selection.any():
        return -200.0
    return float(10.0 * np.log10(max(float(np.mean(power[selection])), 1e-20)))


def _flatness_db(power: np.ndarray, selection: np.ndarray) -> float:
    """Spread of sub-band levels across a region, in dB.

    Measured on sub-bands rather than raw bins on purpose. A single
    periodogram bin fluctuates by about 5.6 dB regardless of the signal, so a
    per-bin standard deviation would mostly report how much averaging the
    caller happened to do. Sub-band means average that away and still catch
    the thing that matters, which is a slope across the region.
    """
    values = power[selection]
    if values.size < _HD_FLATNESS_BANDS:
        return 0.0
    usable = values.size - (values.size % _HD_FLATNESS_BANDS)
    bands = values[:usable].reshape(_HD_FLATNESS_BANDS, -1).mean(axis=1)
    return float(np.std(10.0 * np.log10(np.maximum(bands, 1e-20))))


def detect_hd_radio(
    spectrum_db: np.ndarray,
    bin_width_hz: float,
    carrier_offset_hz: float = 0.0,
) -> HdRadio:
    """Decide whether an FM station at `carrier_offset_hz` also carries HD.

    `spectrum_db` is an fftshifted dBFS spectrum as `psd.Spectrum.process`
    returns it, and `carrier_offset_hz` locates the analog carrier within it -
    zero when the dongle is tuned straight at the station.

    Raises `ValueError` if the spectrum does not reach far enough either side
    of the carrier to see both the sidebands and the noise floor past them.
    Returning "no HD" in that case would make the answer depend on where the
    station happened to fall in the sweep window, which is worse than refusing.
    """
    offsets = _offsets(spectrum_db.size, bin_width_hz) - carrier_offset_hz
    if offsets.size == 0 or offsets[0] > -HD_REFERENCE_HI_HZ or (
        offsets[-1] < HD_REFERENCE_HI_HZ
    ):
        raise ValueError(
            f"spectrum spans {offsets[0] / 1e3:.0f}..{offsets[-1] / 1e3:.0f} kHz "
            f"around the carrier; HD detection needs at least "
            f"+/-{HD_REFERENCE_HI_HZ / 1e3:.0f} kHz"
        )

    power = 10.0 ** (np.asarray(spectrum_db, dtype=np.float64) / 10.0)
    magnitude = np.abs(offsets)

    lower = (offsets <= -HD_SIDEBAND_LO_HZ) & (offsets >= -HD_SIDEBAND_HI_HZ)
    upper = (offsets >= HD_SIDEBAND_LO_HZ) & (offsets <= HD_SIDEBAND_HI_HZ)
    reference = (magnitude >= HD_REFERENCE_LO_HZ) & (magnitude <= HD_REFERENCE_HI_HZ)
    core = magnitude <= HD_ANALOG_CORE_HZ

    noise_db = _mean_power_db(power, reference)
    lower_db = _mean_power_db(power, lower)
    upper_db = _mean_power_db(power, upper)
    core_db = _mean_power_db(power, core)

    lower_snr = lower_db - noise_db
    upper_snr = upper_db - noise_db
    sideband_db = _mean_power_db(power, lower | upper)
    flatness = max(_flatness_db(power, lower), _flatness_db(power, upper))

    present = (
        lower_snr >= HD_MIN_SNR_DB
        and upper_snr >= HD_MIN_SNR_DB
        and abs(lower_snr - upper_snr) <= HD_MAX_IMBALANCE_DB
        and flatness <= HD_MAX_FLATNESS_DB
        # Sanity: the digital part is always well below the analog carrier. If
        # it is not, this is not a hybrid FM station.
        and sideband_db < core_db
    )
    return HdRadio(
        present=present,
        lower_snr_db=lower_snr,
        upper_snr_db=upper_snr,
        level_dbc=sideband_db - core_db,
        flatness_db=flatness,
    )


# -- Shape features for the classifier --------------------------------------
#
# Both of these are deliberately scale-free: they describe how a signal's power
# is arranged across its own bandwidth, not how strong it is. A weak FM station
# and a strong one are the same kind of thing and must classify the same way.


def carrier_fraction(spectrum_db: np.ndarray) -> float:
    """Share of the signal's power sitting in its single strongest bin, 0 to 1.

    This is what separates "has a carrier" from "does not". AM, CW and beacons
    put a large slice of their power into one discrete frequency; FM and every
    digital mode spread it across the whole channel. Expressed as a fraction
    rather than a ratio so the number does not drift with the width of the
    slice being measured.
    """
    if spectrum_db.size == 0:
        return 0.0
    power = 10.0 ** (np.asarray(spectrum_db, dtype=np.float64) / 10.0)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power.max() / total)


def spectral_flatness(spectrum_db: np.ndarray) -> float:
    """Geometric mean over arithmetic mean of power, 0 to 1.

    Near 1 the signal is as featureless as noise across its band, which is
    what a dense digital mode looks like - the same property the HD Radio
    sideband test keys on. Near 0 the power is concentrated in a few bins, so
    there is a carrier or a strong tone. Analog voice lands in between.
    """
    if spectrum_db.size == 0:
        return 0.0
    power = np.maximum(10.0 ** (np.asarray(spectrum_db, dtype=np.float64) / 10.0), 1e-20)
    arithmetic = float(power.mean())
    if arithmetic <= 0.0:
        return 0.0
    # Via the mean of the logs, because the product of thousands of tiny
    # numbers underflows to zero long before it becomes a geometric mean.
    geometric = float(np.exp(np.mean(np.log(power))))
    return float(min(1.0, geometric / arithmetic))


__all__ = [
    "HD_MAX_FLATNESS_DB",
    "HD_MIN_SNR_DB",
    "HD_SIDEBAND_HI_HZ",
    "HD_SIDEBAND_LO_HZ",
    "HdRadio",
    "carrier_fraction",
    "detect_hd_radio",
    "spectral_flatness",
]
