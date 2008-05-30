#ifndef __CLASSIAS_MAXENT_H__
#define __CLASSIAS_MAXENT_H__

#include <float.h>
#include <cmath>
#include <ctime>
#include <iostream>

#include "lbfgs.h"

namespace classias
{

/**
 * Training a log-linear model using the maximum entropy framework.
 *  @param  feature_base        Feature class.
 *  @param  attributes_base     Content class.
 *  @param  instance_base       Instance class.
 *  @param  data_base           Data class.
 *  @param  value_type          
 */
template <class data_iterator_base>
class trainer_maxent : public lbfgs_solver
{
public:
    ///
    typedef double value_type;

    /// A type representing this class.
    typedef trainer_maxent<data_iterator_base> this_class;
    /// A type representing a data_base for training.
    typedef data_iterator_base data_iterator_type;
    /// 
    typedef typename data_iterator_type::value_type instance_type;
    typedef typename instance_type::candidate_type candidate_type;

    /// An array [K] of observation expectations.
    value_type *m_oexps;
    /// An array [K] of model expectations.
    value_type *m_mexps;
    /// An array [K] of feature weights.
    value_type *m_weights;
    /// An array [M] of scores for candidate labels.
    value_type *m_scores;

    /// A group number for holdout evaluation.
    int m_holdout;
    /// Maximum number of iterations.
    int m_maxiter;
    /// Epsilon.
    value_type m_epsilon;
    /// L1-regularization constant.
    value_type m_c1;
    /// L2-regularization constant.
    value_type m_c2;

    /// A data_base for training.
    data_iterator_type* m_begin;
    data_iterator_type* m_end;
    std::ostream* m_os;
    clock_t m_clk_prev;

public:
    trainer_maxent()
    {
        m_oexps = 0;
        m_mexps = 0;
        m_weights = 0;
        m_scores = 0;

        m_holdout = -1;
        m_maxiter = 1000;
        m_epsilon = 1e-5;
        m_c1 = 0;
        m_c2 = 0;

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

        m_begin = NULL;
        m_end = NULL;
        m_os = NULL;
    }

    bool set(const std::string& name, int value)
    {
        if (name == "holdout") {
            m_holdout = value;
        } else if (name == "maxiter") {
            m_maxiter = value;
        } else {
            return false;
        }
        return true;
    }

    bool set(const std::string& name, double value)
    {
        if (name == "epsilon") {
            m_epsilon = value;
        } else if (name == "sigma1") {
            m_c1 = (value <= 0.) ? 0. : 1.0 / value;
        } else if (name == "sigma2") {
            m_c2 = (value <= 0.) ? 0. : 1.0 / (value * value);
        } else {
            return false;
        }
        return true;
    }

    bool set(const std::string& name, const std::string& value)
    {
        return false;
    }

    const value_type* get_weights() const
    {
        return m_weights;
    }

    virtual value_type lbfgs_evaluate(
        const value_type *x, value_type *g, const int n, const value_type step)
    {
        int i;
        value_type loss = 0, norm = 0;
        data_iterator_type begin = *m_begin;
        data_iterator_type end = *m_end;

        // Initialize the model expectations as zero.
        for (i = 0;i < n;++i) {
            m_mexps[i] = 0.;
        }

        // For each instance in the data.
        for (data_iterator_type iti = begin;iti != end;++iti) {
            value_type logp = 0.;
            value_type norm = 0.;
            const instance_type& inst = *iti;
            typename instance_type::const_iterator itc;

            // Exclude instances for holdout evaluation.
            if (inst.group == m_holdout) {
                continue;
            }

            // Compute score[i] for each candidate #i.
            for (i = 0, itc = inst.begin();itc != inst.end();++i, ++itc) {
                m_scores[i] = itc->inner_product(x);
                if (itc->is_true()) logp = m_scores[i];
                norm = logsumexp(norm, m_scores[i], (i == 0));
           }

            // Accumulate the model expectations of attributes.
            for (i = 0, itc = inst.begin();itc != inst.end();++i, ++itc) {
                itc->add(m_mexps, std::exp(m_scores[i] - norm));
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
            for (int i = 0;i < n;++i) {
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

        // Check for the maximum number of iterations.
        if (m_maxiter < k) {
            return 1;
        }

        return 0;
    }

    int train(
        data_iterator_type& begin,
        data_iterator_type& end, std::ostream& os, int holdout = -1)
    {
        size_t M = 0;
        const size_t K = count_attributes(begin, end);

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

        // Report the training parameters.
        os << "Training a maximum entropy model" << std::endl;
        if (m_c1 != 0.) {
            os << "L1 regularization: " << m_c1 << std::endl;
        }
        if (m_c2 != 0.) {
            os << "L2 regularization: " << m_c2 << std::endl;
        }
        if (0 <= m_holdout) {
            os << "Holdout group: " << (m_holdout+1) << std::endl;
        }
        os << std::endl;

        // Compute observation expectations of the features.
        for (data_iterator_type iti = begin;iti != end;++iti) {
            // Skip instances for holdout evaluation.
            if (iti->group == m_holdout) {
                continue;
            }

            // Compute the observation expectations.
            const instance_type& inst = *iti;
            typename instance_type::const_iterator itc;
            for (itc = inst.begin();itc != inst.end();++itc) {
                if (itc->is_true()) {
                    // m_oexps[k] += 1.0 * (*itc)[k].
                    itc->add(m_oexps, 1.0);
                }
            }

            // Store the maximum number of candidates.
            if (M < inst.size()) {
                M = inst.size();
            }
        }

        // Call the L-BFGS solver.
        m_os = &os;
        m_begin = &begin;
        m_end = &end;
        m_clk_prev = clock();
        m_scores = new double[M];
        int ret = lbfgs_solve((const int)K, m_weights, NULL, m_epsilon, m_c1);

        // Report the result from the L-BFGS solver.
        if (ret == 0) {
            os << "L-BFGS resulted in convergence" << std::endl;
        } else {
            os << "L-BFGS terminated with error code (" << ret << ")" << std::endl;
        }

        return ret;
    }

    void holdout_evaluation()
    {
        std::ostream& os = *m_os;
        int num_total = 0;
        int num_tp = 0, num_fp = 0;
        int num_tn = 0, num_fn = 0;
        data_iterator_type begin = *m_begin;
        data_iterator_type end = *m_end;

        // Loop over instances.
        for (data_iterator_type iti = begin;iti != end;++iti) {
            const instance_type& inst = *iti;

            // Exclude instances for holdout evaluation.
            if (inst.group != m_holdout) {
                continue;
            }

            // Compute the score for each candidate #i.
            int i;
            value_type score_max = -DBL_MAX;
            typename instance_type::const_iterator itc;
            typename instance_type::const_iterator itc_max = inst.end();
            typename instance_type::const_iterator itc_ref = inst.end();
            for (i = 0, itc = inst.begin();itc != inst.end();++i, ++itc) {
                value_type score = itc->inner_product(m_weights);

                // Store the candidate that yields the maximum score.
                if (score_max < score) {
                    score_max = score;
                    itc_max = itc;
                }

                // Store the candidate with the reference label.
                if (itc->is_true()) {
                    itc_ref = itc;
                }
            }

            // Update the 2x2 confusion matrix.
            if (itc_max->is_true()) {
                if (itc_max->is_positive()) {
                    ++num_tp;
                } else {
                    ++num_tn;
                }
            } else {
                if (itc_max->is_positive()) {
                    ++num_fp;
                } else {
                    ++num_fn;
                }
            }
            ++num_total;
        }

        // Report accuracy, precision, recall, and f1 score.
        double accuracy = 0.;
        double precision = 0., recall = 0., fscore = 0.;
        if (0 < num_total) {
            accuracy = (num_tp + num_tn) / (double)num_total;
        }
        if (0 < num_tp + num_fp) {
            precision = num_tp / (double)(num_tp + num_fp);
        }
        if (0 < num_tp + num_fn) {
            recall = num_tp / (double)(num_tp + num_fn);
        }
        if (0 < precision + recall) {
            fscore = 2 * precision * recall / (precision + recall);
        }
        os << "Accuracy: " << accuracy << " (" << (num_tp + num_tn) << "/" << num_total << ")" << std::endl;
        os << "Precision: " << precision << " (" << num_tp << "/" << (num_tp + num_fp) << ")" << std::endl;
        os << "Recall: " << recall << " (" << num_tp << "/" << (num_tp + num_fn) << ")" << std::endl;
        os << "F1 score: " << fscore << std::endl;
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

    static int count_attributes(data_iterator_type begin, data_iterator_type end)
    {
        int K = 0;
        for (data_iterator_type iti = begin;iti != end;++iti) {
            const instance_type& inst = *iti;
            typename instance_type::const_iterator itc;
            for (itc = inst.begin();itc != inst.end();++itc) {
                const candidate_type& cand = *itc;
                typename candidate_type::const_iterator ita;
                for (ita = cand.begin();ita != cand.end();++ita) {
                    if (K < ita->first) {
                        K = ita->first;
                    }
                }
            }
        }
        return K+1;
    }
};

};

#endif/*__CLASSIAS_MAXENT_H__*/
