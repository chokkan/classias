/*
 *		Training Maximum Entropy models with L-BFGS.
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

/* $Id:$ */

#ifndef __CLASSIAS_MAXENT_H__
#define __CLASSIAS_MAXENT_H__

#include <float.h>
#include <cmath>
#include <ctime>
#include <iostream>

#include "lbfgs.h"
#include "evaluation.h"
#include "parameters.h"

namespace classias
{

/**
 * Training a log-linear model using the maximum entropy modeling.
 *  @param  data_tmpl           Training data class.
 *  @param  value_tmpl          The type for computation.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_maxent : public lbfgs_solver
{
protected:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A type representing values for internal computations.
    typedef value_tmpl value_type;
    /// A synonym of this class.
    typedef trainer_maxent<data_type, value_type> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// A type representing a candidate for an instance.
    typedef typename instance_type::candidate_type candidate_type;
    /// A type representing a label.
    typedef typename instance_type::label_type label_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

    /// An array [K] of observation expectations.
    value_type *m_oexps;
    /// An array [K] of model expectations.
    value_type *m_mexps;
    /// An array [K] of feature weights.
    value_type *m_weights;
    /// An array [M] of scores for candidate labels.
    value_type *m_scores;

    /// A data set for training.
    const data_type* m_data;
    /// A group number used for holdout evaluation.
    int m_holdout;

    /// Parameters interface.
    parameter_exchange m_params;
    /// Regularization type.
    std::string m_regularization;
    /// Regularization sigma;
    value_type m_regularization_sigma;
    /// Regularization start index.
    int m_regularization_end;
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

    /// L1-regularization constant.
    value_type m_c1;
    /// L2-regularization constant.
    value_type m_c2;

    /// An output stream to which this object outputs log messages.
    std::ostream* m_os;
    /// An internal variable (previous timestamp).
    clock_t m_clk_prev;

public:
    trainer_maxent()
    {
        m_oexps = 0;
        m_mexps = 0;
        m_weights = 0;
        m_scores = 0;

        clear();
    }

    virtual ~trainer_maxent()
    {
        clear();
    }

    void clear()
    {
        delete[] m_weights;
        delete[] m_mexps;
        delete[] m_oexps;
        delete[] m_scores;
        m_oexps = 0;
        m_mexps = 0;
        m_weights = 0;
        m_scores = 0;

        m_data = NULL;
        m_os = NULL;
        m_holdout = -1;

        // Initialize the parameters.
        m_params.init("regularization", &m_regularization, "L2",
            "Regularization method (prior):\n"
            "{'': no regularization, 'L1': L1-regularization, 'L2': L2-regularization}");
        m_params.init("regularization.sigma", &m_regularization_sigma, 5.0,
            "Regularization coefficient (sigma).");
        m_params.init("regularization.end", &m_regularization_end, -1,
            "The index number of features at which L1/L2 norm computations stop. A negative\n"
            "value computes L1/L2 norm with all features.");
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

    const value_type* get_weights() const
    {
        return m_weights;
    }

    virtual value_type lbfgs_evaluate(
        const value_type *x,
        value_type *g,
        const int n,
        const value_type step
        )
    {
        int i;
        value_type loss = 0, norm = 0;
        const data_type& data = *m_data;

        // Initialize the model expectations as zero.
        for (i = 0;i < n;++i) {
            m_mexps[i] = 0.;
        }

        // For each instance in the data.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            value_type logp = 0.;
            value_type norm = 0.;
            typename instance_type::const_iterator itc;

            // Exclude instances for holdout evaluation.
            if (iti->get_group() == m_holdout) {
                continue;
            }

            // Compute score[i] for each candidate #i.
            for (i = 0, itc = iti->begin();itc != iti->end();++i, ++itc) {
                m_scores[i] = itc->inner_product(x);
                if (itc->get_truth()) {
                    logp = m_scores[i];
                }
                norm = logsumexp(norm, m_scores[i], (i == 0));
            }

            // Accumulate the model expectations of features.
            for (i = 0, itc = iti->begin();itc != iti->end();++i, ++itc) {
                itc->add_to(m_mexps, std::exp(m_scores[i] - norm));
            }

            // Accumulate the loss for predicting the instance.
            loss -= (logp - norm);
        }

        // Compute the gradients.
        for (int i = 0;i < n;++i) {
            g[i] = -(m_oexps[i] - m_mexps[i]);
        }

        // Apply L2 regularization if necessary.
        if (m_c2 != 0.) {
            value_type norm = 0.;
            for (int i = 0;i < m_regularization_end;++i) {
                g[i] += (m_c2 * x[i]);
                norm += x[i] * x[i];
            }
            loss += (m_c2 * norm * 0.5);
        }

        return loss;
    }

    virtual int lbfgs_progress(
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

    int train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1,
        bool false_analysis = false
        )
    {
        label_type M = 0;
        const size_t K = data.traits.num_features();

        // Initialize feature expectations and weights.
        m_oexps = new double[K];
        m_mexps = new double[K];
        m_weights = new double[K];
        for (size_t k = 0;k < K;++k) {
            m_oexps[k] = 0.;
            m_mexps[k] = 0.;
            m_weights[k] = 0.;
        }
        m_holdout = holdout;

        // Set the internal parameters.
        if (m_regularization == "L1" || m_regularization == "l1") {
            m_c1 = 1.0 / m_regularization_sigma;
            m_c2 = 0.;
            m_lbfgs_linesearch = "Backtracking";
        } else if (m_regularization == "L2" || m_regularization == "l2") {
            m_c1 = 0.;
            m_c2 = 1.0 / (m_regularization_sigma * m_regularization_sigma);
        } else {
            m_c1 = 0.;
            m_c2 = 0.;
        }

        // Set the default value of m_regularization_end.
        if (m_regularization_end < 0) {
            m_regularization_end = K;
        }

        // Report the training parameters.
        os << "Training a maximum entropy model" << std::endl;
        m_params.show(os);
        os << std::endl;

        // Compute observation expectations of the features.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            // Skip instances for holdout evaluation.
            if (iti->get_group() == m_holdout) {
                continue;
            }

            // Compute the observation expectations.
            typename instance_type::const_iterator itc;
            for (itc = iti->begin();itc != iti->end();++itc) {
                if (itc->get_truth()) {
                    // m_oexps[k] += 1.0 * (*itc)[k].
                    itc->add_to(m_oexps, 1.0);
                }
            }

            if (M < (label_type)iti->size()) {
                M = (label_type)iti->size();
            }
        }

        // Initialze the variables used by callback functions.
        m_os = &os;
        m_data = &data;
        m_clk_prev = clock();
        m_scores = new double[M];

        // Call the L-BFGS solver.
        int ret = lbfgs_solve(
            (const int)K,
            m_weights,
            NULL,
            m_lbfgs_num_memories,
            m_lbfgs_epsilon,
            m_lbfgs_stop,
            m_lbfgs_delta,
            m_lbfgs_maxiter,
            m_lbfgs_linesearch,
            m_lbfgs_max_linesearch,
            m_c1,
            m_regularization_end
            );

        // Report the result from the L-BFGS solver.
        lbfgs_output_status(os, ret);

        return ret;
    }

    void holdout_evaluation()
    {
        std::ostream& os = *m_os;
        const data_type& data = *m_data;
        accuracy acc;
        confusion_matrix matrix(data.labels.size());

        // Loop over instances.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            // Exclude instances for holdout evaluation.
            if (iti->get_group() != m_holdout) {
                continue;
            }

            // Compute the score for each candidate #i.
            label_type true_label = -1;
            value_type score_max = -DBL_MAX;
            typename instance_type::const_iterator itc;
            typename instance_type::const_iterator itc_max = iti->end();
            for (itc = iti->begin();itc != iti->end();++itc) {
                value_type score = itc->inner_product(m_weights);

                // Store the candidate that yields the maximum score.
                if (score_max < score) {
                    score_max = score;
                    itc_max = itc;
                }

                // Store the reference label.
                if (itc->get_truth()) {
                    true_label = itc->get_label();
                }
            }

            // Update the accuracy.
            acc.set(itc_max->get_truth());

            // Update the confusion matrix.
            if (true_label != -1) {
                matrix(true_label, itc_max->get_label())++;
            }
        }

        // Report accuracy, precision, recall, and f1 score.
        acc.output(os);
        matrix.output_micro(os, data.positive_labels.begin(), data.positive_labels.end());
    }

protected:
    static value_type logsumexp(value_type x, value_type y, int flag)
    {
        value_type vmin, vmax;

        if (flag) return y;
        if (x == y) return x + 0.69314718055;   /* log(2) */
        if (x < y) {
            vmin = x; vmax = y;
        } else {
            vmin = y; vmax = x;
        }
        if (vmin + 50 < vmax)
            return vmax;
        else
            return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
};

};

#endif/*__CLASSIAS_MAXENT_H__*/
