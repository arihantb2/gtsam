/**
 * @file  ImuProcessNoise.h
 * @brief Shared IMU process-noise PSD helpers for tg_eqf / tfg_inekf / mekf
 *        scenario runners.
 *
 * Runners build continuous-time PSDs (variances) from the same CLI sigmas,
 * optionally apply a shared numerical floor on ideal-IMU runs, and convert to
 * each filter's ImuNoise type. Mapping into the covariance (lift B-matrix vs
 * diagonal Qc) stays inside each filter.
 */
#pragma once

#include <cmath>

namespace imu_scenarios {

/// Continuous-time IMU noise PSDs (per-axis variances), isotropic per channel.
struct ImuNoisePSD {
  double gyro = 0.0;      ///< gyro white-noise PSD      (rad^2/s)
  double accel = 0.0;     ///< accelerometer white-noise PSD (m^2/s^3)
  double gyro_rw = 0.0;   ///< gyro bias random-walk PSD   (rad^2/s^3)
  double accel_rw = 0.0;  ///< accel bias random-walk PSD  (m^2/s^5)
};

/// Build PSDs from continuous noise densities: PSD = sigma^2 when sigma > 0.
template <typename RunOptions>
inline ImuNoisePSD imuNoiseFromOptions(const RunOptions& opts) {
  ImuNoisePSD n;
  if (opts.gyro_noise_sigma > 0.0) {
    n.gyro = opts.gyro_noise_sigma * opts.gyro_noise_sigma;
  }
  if (opts.accel_noise_sigma > 0.0) {
    n.accel = opts.accel_noise_sigma * opts.accel_noise_sigma;
  }
  if (opts.gyro_bias_rw > 0.0) {
    n.gyro_rw = opts.gyro_bias_rw * opts.gyro_bias_rw;
  }
  if (opts.accel_bias_rw > 0.0) {
    n.accel_rw = opts.accel_bias_rw * opts.accel_bias_rw;
  }
  return n;
}

inline bool isZeroProcessNoise(const ImuNoisePSD& n) {
  return n.gyro == 0.0 && n.accel == 0.0 && n.gyro_rw == 0.0 &&
         n.accel_rw == 0.0;
}

/// Small continuous-time floor used when the simulated IMU is ideal (all CLI
/// sigmas zero) so covariance propagation stays numerically regularized.
inline ImuNoisePSD defaultProcessNoiseFloor() {
  ImuNoisePSD n;
  n.gyro = 1e-4;
  n.accel = 1e-3;
  n.gyro_rw = 1e-6;
  n.accel_rw = 1e-5;
  return n;
}

inline ImuNoisePSD applyProcessNoiseFloor(const ImuNoisePSD& psd,
                                          const ImuNoisePSD& floor) {
  ImuNoisePSD out = psd;
  if (out.gyro == 0.0) {
    out.gyro = floor.gyro;
  }
  if (out.accel == 0.0) {
    out.accel = floor.accel;
  }
  if (out.gyro_rw == 0.0) {
    out.gyro_rw = floor.gyro_rw;
  }
  if (out.accel_rw == 0.0) {
    out.accel_rw = floor.accel_rw;
  }
  return out;
}

/// PSDs from CLI; when every channel is unset, apply
/// defaultProcessNoiseFloor().
template <typename RunOptions>
inline ImuNoisePSD resolvedProcessNoise(const RunOptions& opts) {
  const ImuNoisePSD from_opts = imuNoiseFromOptions(opts);
  if (isZeroProcessNoise(from_opts)) {
    return defaultProcessNoiseFloor();
  }
  return from_opts;
}

/// Copy PSD fields into a filter-specific ImuNoise struct (identical layout).
template <typename FilterImuNoise>
inline FilterImuNoise toFilterImuNoise(const ImuNoisePSD& psd) {
  FilterImuNoise n;
  n.gyro = psd.gyro;
  n.accel = psd.accel;
  n.gyro_rw = psd.gyro_rw;
  n.accel_rw = psd.accel_rw;
  return n;
}

}  // namespace imu_scenarios
