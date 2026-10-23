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
 * Simple Kalman Filter for variable gain low-passing
 *
 * @author Nicolas de Palezieux (Sunflower Labs) <ndepal@gmail.com>
 *
 */

#include "KalmanFilter.h"

namespace landing_target_estimator
{
using namespace typed;

KalmanFilter::KalmanFilter(matrix::Vector<float, 2> &initial, matrix::Matrix<float, 2, 2> &covInit)
{
	init(initial, covInit);
}

void KalmanFilter::init(matrix::Vector<float, 2> &initial, matrix::Matrix<float, 2, 2> &covInit)
{
	// the values in the units of the elements: m, m/s, and their products
	_filter.x(State{fcarouge_px4::Matrix<float, 2, 1>{initial}});
	_filter.p(Covariance{fcarouge_px4::Matrix<float, 2, 2>{covInit}});
}

void KalmanFilter::init(float initial0, float initial1, float covInit00, float covInit11)
{
	_filter.x(State{initial0 * metre, initial1 * metre_per_second});
	Covariance covInit{};
	covInit.at<0, 0>(covInit00 * (metre * metre));
	covInit.at<1, 1>(covInit11 * (metre_per_second * metre_per_second));
	_filter.p(covInit);
}

void KalmanFilter::predict(float dt, float acc, float acc_unc)
{
	_filter.predict(dt * second, acc_unc * AccelerationVariance::reference, Input{acc * metre_per_second_squared});
}

bool KalmanFilter::update(float meas, float measUnc)
{
	const Position position{meas * metre};
	const PositionVariance variance{measUnc *(metre * metre)};

	// H = [1, 0]
	_residual = position - _filter.x().at<0>();

	// H * P * H^T simply selects P(0,0)
	_innovCov = _filter.p().at<0, 0>() + variance;

	// outlier rejection, before the filter's update
	const float beta = (_residual / _innovCov * _residual).numerical_value_in(mp_units::one);

	// 5% false alarm probability
	if (beta > 3.84f) {
		return false;
	}

	_filter.r(OutputVariance{variance});
	_filter.update(Output{position});

	return true;
}

void KalmanFilter::getState(matrix::Vector<float, 2> &state)
{
	state = _filter.x().data().toMatrix();
}

void KalmanFilter::getState(float &state0, float &state1)
{
	state0 = _filter.x().at<0>().numerical_value_in(metre);
	state1 = _filter.x().at<1>().numerical_value_in(metre_per_second);
}

void KalmanFilter::getCovariance(matrix::Matrix<float, 2, 2> &covariance)
{
	covariance = _filter.p().data().toMatrix();
}

void KalmanFilter::getCovariance(float &cov00, float &cov11)
{
	cov00 = _filter.p().at<0, 0>().numerical_value_in(metre * metre);
	cov11 = _filter.p().at<1, 1>().numerical_value_in(metre_per_second * metre_per_second);
}

void KalmanFilter::getInnovations(float &innov, float &innovCov)
{
	innov = _residual.numerical_value_in(metre);
	innovCov = _innovCov.numerical_value_in(metre * metre);
}

} // namespace landing_target_estimator
