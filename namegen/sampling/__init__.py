"""Frequency-calibrated sampling helpers.

M1 ships a no-op placeholder. Real Zipf-correct resampling against era
frequency tables arrives with M7.
"""

from namegen.sampling.frequency import FrequencyCalibrator

__all__ = ["FrequencyCalibrator"]
