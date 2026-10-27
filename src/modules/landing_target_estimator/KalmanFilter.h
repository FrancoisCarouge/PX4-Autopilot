/****************************************************************************
 *
 *   Copyright (c) 2013-2018 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/*
 * @file KalmanFilter.h
 * Simple Kalman Filter for variable gain low-passing
 *
 * Constant velocity model. Prediction according to
 * x_{k+1} = A * x_{k}
 * with A = [1 dt; 0 1]
 *
 * Update with a direct measurement of the first state:
 * H = [1 0]
 *
 * @author Nicolas de Palezieux (Sunflower Labs) <ndepal@gmail.com>
 *
 */

#pragma once

#include <matrix/math.hpp>
#include <mathlib/mathlib.h>

#include <fcarouge/kalman.hpp>
#include <fcarouge/px4_typed_linear_algebra.hpp>
#include <typed/fcarouge/kalman_typed_linear_algebra.hpp>
#include <mp_units/fcarouge/kalman_mp_units.hpp>
#include <mp-units/systems/si.h>

namespace landing_target_estimator
{
namespace typed
{
template <auto Reference>
using Quantity = mp_units::quantity<Reference, float>;

inline constexpr auto metre = mp_units::si::metre;
inline constexpr auto second = mp_units::si::second;
inline constexpr auto metre_per_second = metre / second;
inline constexpr auto metre_per_second_squared = metre / (second *second);

using Seconds = Quantity<second>;
using Position = Quantity<metre>;
using PositionVariance = Quantity<metre *metre>;
using Velocity = Quantity<metre_per_second>;
using Acceleration = Quantity<metre_per_second_squared>;
using AccelerationVariance = Quantity<metre_per_second_squared *metre_per_second_squared>;

/// the position and velocity of the target relative to the vehicle, along one axis
using State = fcarouge_px4::ColumnVector<float, Position, Velocity>;
/// the measured relative position
using Output = fcarouge_px4::ColumnVector<float, Position>;
/// the acceleration of the target relative to the vehicle
using Input = fcarouge_px4::ColumnVector<float, Acceleration>;

namespace kf = fcarouge::kalman_filter::internal;
using Covariance = kf::evaluate<kf::product<State, kf::evaluate<kf::transpose<State>>>>;
using OutputVariance = kf::evaluate<kf::product<Output, kf::evaluate<kf::transpose<Output>>>>;
using StateTransition = kf::evaluate<kf::quotient<State, State>>;
using InputControl = kf::evaluate<kf::quotient<State, Input>>;

/// G = [dt^2 / 2, dt]: the change of the state per acceleration
inline InputControl inputControl(const Seconds &dt)
{
	return InputControl{dt *dt / 2.f, dt};
}

/*
 * x[k+1] = A x[k] + G u[k] with A = [1 dt; 0 1], the acceleration u of
 * variance acc_unc: Q = G G^T acc_unc. The output is the position: H = [1 0],
 * the library's default output model.
 */
inline auto makeFilter()
{
	using namespace fcarouge;
	return kalman{
		state{State{0.f * metre, 0.f * metre_per_second}},
		output<Output>,
		input<Input>,
		// set by init()
		estimate_uncertainty{Covariance{}},
		process_uncertainty{
			[](const State &, const Seconds & dt, const AccelerationVariance & acc_unc) -> Covariance {
				const InputControl g{inputControl(dt)};
				return Covariance{g *acc_unc * transposed(g)};
			}},
		// set by update()
		output_uncertainty{OutputVariance{0.f *(metre * metre)}},
		state_transition{[](const Input &, const Seconds & dt, const AccelerationVariance &) -> StateTransition {
				StateTransition a{};
				a.at<0, 0>(1.f * mp_units::one);
				a.at<0, 1>(dt);
				a.at<1, 1>(1.f * mp_units::one);
				return a;
			}},
		input_control{[](const Seconds & dt, const AccelerationVariance &) -> InputControl {
				return inputControl(dt);
			}},
		prediction_types<Seconds, AccelerationVariance>,
		// the covariance corrected as (I - K H) P, as the original
		simple_form};
}
} // namespace typed

class KalmanFilter
{
public:
	/**
	 * Default constructor, state not initialized
	 */
	KalmanFilter() {};

	/**
	 * Constructor, initialize state
	 */
	KalmanFilter(matrix::Vector<float, 2> &initial, matrix::Matrix<float, 2, 2> &covInit);

	/**
	 * Default desctructor
	 */
	virtual ~KalmanFilter() {};

	/**
	 * Initialize filter state
	 * @param initial initial state
	 * @param covInit initial covariance
	 */
	void init(matrix::Vector<float, 2> &initial, matrix::Matrix<float, 2, 2> &covInit);

	/**
	 * Initialize filter state, only specifying diagonal covariance elements
	 * @param initial0  first initial state
	 * @param initial1  second initial state
	 * @param covInit00 initial variance of first state
	 * @param covinit11 initial variance of second state
	 */
	void init(float initial0, float initial1, float covInit00, float covInit11);

	/**
	 * Predict the state with an external acceleration estimate
	 * @param dt            Time delta in seconds since last state change
	 * @param acc           Acceleration estimate
	 * @param acc_unc       Variance of acceleration estimate
	 */
	void predict(float dt, float acc, float acc_unc);

	/**
	 * Update the state estimate with a measurement
	 * @param meas    state measeasurement
	 * @param measUnc measurement uncertainty
	 * @return update success (measurement not rejected)
	 */
	bool update(float meas, float measUnc);

	/**
	 * Get the current filter state
	 * @param x1 State
	 */
	void getState(matrix::Vector<float, 2> &state);

	/**
	 * Get the current filter state
	 * @param state0 First state
	 * @param state1 Second state
	 */
	void getState(float &state0, float &state1);

	/**
	 * Get state covariance
	 * @param covariance Covariance of the state
	 */
	void getCovariance(matrix::Matrix<float, 2, 2> &covariance);

	/**
	 * Get state variances (diagonal elements)
	 * @param cov00 Variance of first state
	 * @param cov11 Variance of second state
	 */
	void getCovariance(float &cov00, float &cov11);

	/**
	 * Get measurement innovation and covariance of last update call
	 * @param innov Measurement innovation
	 * @param innovCov Measurement innovation covariance
	 */
	void getInnovations(float &innov, float &innovCov);

private:
	/// the state and its covariance
	decltype(typed::makeFilter()) _filter{typed::makeFilter()};

	typed::Position _residual{}; // residual of last measurement update

	typed::PositionVariance _innovCov{}; // innovation covariance of last measurement update
};
} // namespace landing_target_estimator
