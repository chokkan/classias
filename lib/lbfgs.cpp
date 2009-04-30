/*
 *		C++ driver for liblbfgs.
 *
 * Copyright (c) 2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Northwestern University, University of Tokyo,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#include <iostream>
#include <classias/base.h>
#include <classias/lbfgs.h>
#include <lbfgs.h>

namespace classias
{

static double
__lbfgs_evaluate(
    void *inst, const double *x, double *g, const int n, const double step)
{
    lbfgs_solver* pt = reinterpret_cast<lbfgs_solver*>(inst);
    return pt->lbfgs_evaluate(x, g, n, step);
}

static int
__lbfgs_progress(
    void *inst,
    const double *x, const double *g, const double fx,
    const double xnorm, const double gnorm,
    const double step,
    int n, int k, int ls
    )
{
    lbfgs_solver* pt = reinterpret_cast<lbfgs_solver*>(inst);
    int ret = pt->lbfgs_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    if (ret != 0) {
        return LBFGSERR_MAXIMUMITERATION;
    }
    return 0;
}

int lbfgs_solver::lbfgs_solve(
    const int n,
    double *x,
    double *ptr_fx,
    int m,
    double epsilon,
    int stop,
    double delta,
    int maxiter,
    std::string linesearch,
    int maxlinesearch,
    double c1,
    int l1_start
    )
{
    // Set L-BFGS parameters.
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.m = m;
    param.epsilon = epsilon;
    param.past = stop;
    param.delta = delta;
    param.max_iterations = maxiter;
    if (linesearch == "Backtracking") {
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    }
    param.max_linesearch = maxlinesearch;
    param.orthantwise_c = c1;
    param.orthantwise_start = l1_start;
    param.orthantwise_end = n;

    // Call L-BFGS routine.
    return lbfgs(
        n,
        x,
        ptr_fx,
        __lbfgs_evaluate,
        __lbfgs_progress,
        this,
        &param
        );
}

void lbfgs_solver::lbfgs_output_status(std::ostream& os, int status)
{
    if (status == LBFGS_CONVERGENCE) {
        os << "L-BFGS resulted in convergence" << std::endl;
    } else if (status == LBFGS_STOP) {
        os << "L-BFGS terminated with the stopping criteria" << std::endl;
    } else {
        os << "L-BFGS terminated with error code (" << status << ")" << std::endl;
    }
}


};
