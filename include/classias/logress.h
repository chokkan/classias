#ifndef __TRANKIN_LOGRESS_H__
#define __TRANKIN_LOGRESS_H__

#include <float.h>
#include <cmath>
#include <ctime>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include "lbfgs.h"
#include "evaluation.h"
#include "parameters.h"

namespace classias {

/**
 * Training a logistic regression model.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_logress : public lbfgs_solver
{
public:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A type representing values for internal computations.
    typedef value_tmpl value_type;
    /// A synonym of this class.
    typedef trainer_logress<data_type, value_type> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// .
    typedef typename instance_type::features_type features_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

    /// An array [K] of feature weights.
    value_type *m_weights;

protected:
    /// A data set for training.
    const data_type* m_data;
    /// A group number for holdout evaluation.
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
    trainer_logress()
    {
        m_weights = NULL;
        clear();
    }

    virtual ~trainer_logress()
    {
        clear();
    }

    void clear()
    {
        delete[] m_weights;
        m_weights = 0;

        m_data = NULL;
        m_os = 0;
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
        value_type loss = 0;
        typename data_type::const_iterator iti;

        // Initialize the gradient of every weight as zero.
        for (int i = 0;i < n;++i) {
            g[i] = 0.;
        }

        // For each instance in the data.
        for (iti = m_data->begin();iti != m_data->end();++iti) {
            value_type z = 0.;
            value_type d = 0.;

            // Exclude instances for holdout evaluation.
            if (iti->get_group() == m_holdout) {
                continue;
            }

            // Compute the instance score.
            z = iti->inner_product(x);

            if (z < -50.) {
                if (iti->get_truth()) {
                    d = 1.;
                    loss -= z;
                } else {
                    d = 0.;
                }
            } else if (50. < z) {
                if (iti->get_truth()) {
                    d = 0.;
                } else {
                    d = -1.;
                    loss += z;
                }
            } else {
                double p = 1.0 / (1.0 + std::exp(-z));
                if (iti->get_truth()) {
                    d = 1.0 - p;
                    loss -= std::log(p);
                } else {
                    d = -p;
                    loss -= std::log(1-p);                
                }
            }

            // Update the gradients for the weights.
            iti->add_to(g, -d);
        }

	    // L2 regularization.
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
        int holdout = -1
        )
    {
        const size_t K = data.num_features();
        typename data_type::const_iterator it;

        // Initialize feature weights.
        m_weights = new double[K];
        for (size_t k = 0;k < K;++k) {
            m_weights[k] = 0.;
        }
        m_holdout = holdout;

        // Set the internal parameters.
        if (m_regularization == "L1" || m_regularization == "l1") {
            m_c1 = 1.0 / m_regularization_sigma;
            m_c2 = 0.;
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

        os << "Training a logistic regression model" << std::endl;
        m_params.show(os);
        os << std::endl;

        // Call the L-BFGS solver.
        m_os = &os;
        m_data = &data;
        m_clk_prev = clock();
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
        int positive_labels[] = {1};
        confusion_matrix matrix(2);

        // For each classification_instance_base in the data_base.
        for (const_iterator iti = m_data->begin();iti != m_data->end();++iti) {
            // Skip instances for training.
            if (iti->get_group() != m_holdout) {
                continue;
            }

            // Compute the logit.
            value_type z = iti->inner_product(m_weights);

            // Classify the instance.
            matrix(
                (iti->get_truth() ? 1 : 0),
                (z <= 0. ? 0 : 1)
                )++;
        }

        matrix.output_accuracy(os);
        matrix.output_micro(os, positive_labels, positive_labels+1);
    }
};

};

#endif/*__TRANKIN_LOGRESS_H__*/
