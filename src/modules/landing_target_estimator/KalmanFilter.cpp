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
 * @file KalmanFilter.cpp
 *
 * @author Nicolas de Palezieux (Sunflower Labs) <ndepal@gmail.com>
 */

#include "KalmanFilter.h"

namespace landing_target_estimator
{

using px4_typed_linear_algebra::Element;

KalmanFilter::Transition KalmanFilter::propagation::operator()(const Input &, const Time &dt,
		const AccelerationVariance &) const
{
	Transition A{};
	A.at<0, 0>(1.f * Element<Transition, 0, 0>::reference);
	A.at<1, 1>(1.f * Element<Transition, 1, 1>::reference);
	A.at<0, 1>(dt);
	return A;
}

KalmanFilter::InputControl KalmanFilter::noise_model::operator()(const Time &dt, const AccelerationVariance &) const
{
	InputControl G{};
	G.at<0>(dt * dt / 2);
	G.at<1>(dt);
	return G;
}

KalmanFilter::Covariance KalmanFilter::process_noise::operator()(const State &, const Time &dt,
		const AccelerationVariance &acc_unc) const
{
	const InputControl G{noise_model{}(dt, acc_unc)};
	return Covariance{G * fcarouge::transposed(G) * acc_unc};
}

void KalmanFilter::init(Length position, Speed velocity, PositionVariance position_var, VelocityVariance velocity_var)
{
	_filter.x(State{position, velocity});
	Covariance covInit{};
	covInit.at<0, 0>(position_var);
	covInit.at<1, 1>(velocity_var);
	_filter.p(covInit);
}

void KalmanFilter::predict(Time dt, Acceleration acc, AccelerationVariance acc_unc)
{
	// x = A x + G acc, P = A P A^T + G G^T acc_unc
	_filter.predict(dt, acc_unc, Input{acc});
}

bool KalmanFilter::update(Length meas, PositionVariance measUnc)
{
	// H = [1, 0]
	_residual = meas - _filter.x().at<0>();

	// H * P * H^T simply selects P(0,0)
	_innovCov = _filter.p().at<0, 0>() + measUnc;

	// outlier rejection
	const auto beta = _residual / _innovCov * _residual;

	// 5% false alarm probability
	if (beta > 3.84f * mp_units::one) {
		return false;
	}

	// K = P H^T / S, x += K y, P = (I - K H) P
	_filter.r(measUnc);
	_filter.update(meas);
	return true;
}

} // namespace landing_target_estimator
