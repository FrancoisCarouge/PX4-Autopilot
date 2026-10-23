############################################################################
#
#   Copyright (c) 2026 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################


# The header-only FrancoisCarouge Kalman and TypedLinearAlgebra libraries,
# with mp-units, as an interface library. Their headers are system headers:
# PX4's warning flags do not apply to them. mp-units is used freestanding, as
# on NuttX: no exceptions, no contract checks, no throwing constraints.
add_library(fcarouge INTERFACE)
target_include_directories(fcarouge SYSTEM INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/Kalman/include
	${CMAKE_CURRENT_SOURCE_DIR}/Kalman/support
	${CMAKE_CURRENT_SOURCE_DIR}/TypedLinearAlgebra/include
	${CMAKE_CURRENT_SOURCE_DIR}/TypedLinearAlgebra/support/mp_units
	${CMAKE_CURRENT_SOURCE_DIR}/mp-units/src/core/include
	${CMAKE_CURRENT_SOURCE_DIR}/mp-units/src/systems/include
)
target_compile_definitions(fcarouge INTERFACE
	MP_UNITS_HOSTED=0
	MP_UNITS_API_CONTRACTS=0
	MP_UNITS_API_THROWING_CONSTRAINTS=0
)
