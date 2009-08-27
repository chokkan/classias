/*
 *		Base class for training a model with L-BFGS.
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
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
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

#ifndef __CLASSIAS_LBFGS_BASE_H__
#define __CLASSIAS_LBFGS_BASE_H__

#include <limits.h>
#include <ctime>
#include <iostream>
#include <lbfgs.h>
#include "../../parameters.h"

namespace classias
{

namespace train
{

template <
    class value_tmpl = double
>
class lbfgs_base
{
public:
    /// A type representing values for internal computations.
    typedef value_tmpl value_type;

protected:
    /// An array [K] of feature weights.
    value_type *m_weights;

    /// Parameter interface.
    parameter_exchange m_params;

    /// Sigma for L1-regularization;
    value_type m_regularization_sigma1;
    /// Sigma for L2-regularization;
    value_type m_regularization_sigma2;
    /// The number of memories in L-BFGS.
    int m_lbfgs_num_memories;
    /// L-BFGS epsilon for convergence.
    value_type m_lbfgs_epsilon;
    /// Number of iterations for stopping criterion.
    int m_lbfgs_stop;
    /// The delta threshold for stopping criterion.
    value_type m_lbfgs_delta;
    /// Maximum number of L-BFGS iterations.
    int m_lbfgs_maxiter;
    /// Line search algorithm.
    std::string m_lbfgs_linesearch;
    /// The maximum number of trials for the line search algorithm.
    int m_lbfgs_max_linesearch;

    /// A group number for holdout evaluation.
    int m_holdout;
    /// An output stream to which this object outputs log messages.
    std::ostream* m_os;

    /// An internal variable (previous timestamp).
    clock_t m_clk_prev;
    /// The start index for regularization.
    int m_regularization_start;

public:
    lbfgs_base()
    {
        m_weights = NULL;
        clear();
    }

    virtual ~lbfgs_base()
    {
        clear();
    }

    void clear()
    {
        // Free the weight vector.
        delete[] m_weights;
        m_weights = NULL;

        // Initialize the members.
        m_holdout = -1;
        m_os = NULL;

        // Initialize the parameters.
        m_params.init("regularization.sigma1", &m_regularization_sigma1, 0.0,
            "Coefficient (sigma) for L1-regularization.");
        m_params.init("regularization.sigma2", &m_regularization_sigma2, 1.0,
            "Coefficient (sigma) for L2-regularization.");
        m_params.init("lbfgs.num_memories", &m_lbfgs_num_memories, 6,
            "The number of corrections to approximate the inverse hessian matrix.");
        m_params.init("lbfgs.epsilon", &m_lbfgs_epsilon, 1e-5,
            "Epsilon for testing the convergence of the log likelihood.");
        m_params.init("lbfgs.stop", &m_lbfgs_stop, 10,
            "The duration of iterations to test the stopping criterion.");
        m_params.init("lbfgs.delta", &m_lbfgs_delta, 1e-5,
            "The threshold for the stopping criterion; an L-BFGS iteration stops when the\n"
            "improvement of the log likelihood over the last ${lbfgs.stop} iterations is\n"
            "no greater than this threshold.");
        m_params.init("lbfgs.max_iterations", &m_lbfgs_maxiter, INT_MAX,
            "The maximum number of L-BFGS iterations.");
        m_params.init("lbfgs.linesearch", &m_lbfgs_linesearch, "MoreThuente",
            "The line search algorithm used in L-BFGS updates:\n"
            "{'MoreThuente': More and Thuente's method, 'Backtracking': backtracking}");
        m_params.init("lbfgs.max_linesearch", &m_lbfgs_max_linesearch, 20,
            "The maximum number of trials for the line search algorithm.");
    }

    parameter_exchange& params()
    {
        return m_params;
    }

    void initialize_weights(const size_t K)
    {
        m_weights = new double[K];
        for (size_t k = 0;k < K;++k) {
            m_weights[k] = 0.;
        }
    }

    const value_type* get_weights() const
    {
        return m_weights;
    }

    static value_type
    __lbfgs_evaluate(
        void *inst, const value_type *x, value_type *g, const int n, const value_type step)
    {
        lbfgs_base* pt = reinterpret_cast<lbfgs_base*>(inst);
        return pt->lbfgs_evaluate(x, g, n, step);
    }

    value_type lbfgs_evaluate(
        const value_type *x,
        value_type *g,
        const int n,
        const value_type step
        )
    {
        // Compute the loss and gradients.
        value_type loss = loss_and_gradient(x, g, n);

	    // L2 regularization.
	    if (m_regularization_sigma2 != 0.) {
            value_type norm = 0.;
            for (int i = m_regularization_start;i < n;++i) {
                g[i] += (m_regularization_sigma2 * x[i]);
                norm += x[i] * x[i];
            }
            loss += (m_regularization_sigma2 * norm * 0.5);
	    }

        return loss;
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
        lbfgs_base* pt = reinterpret_cast<lbfgs_base*>(inst);
        int ret = pt->lbfgs_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
        if (ret != 0) {
            return LBFGSERR_MAXIMUMITERATION;
        }
        return 0;
    }

    int lbfgs_progress(
        const value_type *x,
        const value_type *g,
        const value_type fx,
        const value_type xnorm,
        const value_type gnorm,
        const value_type step,
        int n,
        int k,
        int ls)
    {
        // Compute the duration required for this iteration.
        std::ostream& os = *m_os;
        clock_t duration, clk = std::clock();
        duration = clk - m_clk_prev;
        m_clk_prev = clk;

        // Count the number of active features.
        int num_active = 0;
        for (int i = 0;i < n;++i) {
            if (x[i] != 0.) {
                ++num_active;
            }
        }

        // Output the current progress.
        os << "***** Iteration #" << k << " *****" << std::endl;
        os << "Log-likelihood: " << -fx << std::endl;
        os << "Feature norm: " << xnorm << std::endl;
        os << "Error norm: " << gnorm << std::endl;
        os << "Active features: " << num_active << " / " << n << std::endl;
        os << "Line search trials: " << ls << std::endl;
        os << "Line search step: " << step << std::endl;
        os << "Seconds required for this iteration: " <<
            duration / (double)CLOCKS_PER_SEC << std::endl;
        os.flush();

        // Holdout evaluation if necessary.
        if (m_holdout != -1) {
            holdout_evaluation();
        }

        // Output an empty line.
        os << std::endl;
        os.flush();

        return 0;
    }

    int lbfgs_solve(
        const int K,
        std::ostream& os,
        int holdout,
        int regularization_start
        )
    {
        // Set L-BFGS parameters.
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.m = m_lbfgs_num_memories;
        param.epsilon = m_lbfgs_epsilon;
        param.past = m_lbfgs_stop;
        param.delta = m_lbfgs_delta;
        param.max_iterations = m_lbfgs_maxiter;
        if (m_lbfgs_linesearch == "Backtracking") {
            param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        }
        param.max_linesearch = m_lbfgs_max_linesearch;
        param.orthantwise_c = m_regularization_sigma1;
        param.orthantwise_start = regularization_start;
        param.orthantwise_end = K;

        // Store the start clock.
        m_os = &os;
        m_clk_prev = clock();
        m_holdout = holdout;
        m_regularization_start = regularization_start;

        // Call L-BFGS routine.
        return lbfgs(
            K,
            m_weights,
            NULL,
            __lbfgs_evaluate,
            __lbfgs_progress,
            this,
            &param
            );
    }

    void lbfgs_output_status(std::ostream& os, int status)
    {
        if (status == LBFGS_CONVERGENCE) {
            os << "L-BFGS resulted in convergence" << std::endl;
        } else if (status == LBFGS_STOP) {
            os << "L-BFGS terminated with the stopping criteria" << std::endl;
        } else {
            os << "L-BFGS terminated with error code (" << status << ")" << std::endl;
        }
    }


    virtual value_type loss_and_gradient(
        const value_type *x,
        value_type *g,
        const int n
        ) = 0;

    virtual void holdout_evaluation() = 0;
};

};

};

#endif/*__CLASSIAS_LBFGS_BASE_H__*/
