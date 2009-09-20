/*
 *      Batch scheduler for online training algorithms.
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

#ifndef __CLASSIAS_TRAIN_ONLINE_SCHEDULER_H__
#define __CLASSIAS_TRAIN_ONLINE_SCHEDULER_H__

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <iostream>
#include <iterator>
#include <vector>
#include <classias/parameters.h>
#include <classias/evaluation.h>

namespace classias {

namespace train {

template <class iterator_type>
static iterator_type
random_sample(
    iterator_type first, iterator_type last
    )
{
    int n = (int)std::distance(first, last);
    int i = std::rand() % n;
    std::advance(first, i);
    return first;
}

template <class container_type, class iterator_type>
static void
shuffle_permutation(
    container_type& cont, iterator_type first, iterator_type last
    )
{
    size_t i = 0;
    for (iterator_type it = first;it != last;++it) {
        cont[i++] = it;
    }
    std::random_shuffle(cont.begin(), cont.end());
}

template <class value_type, class iterator_type>
static value_type compute_variance(iterator_type first, iterator_type last, value_type avg)
{
    int n = 0;
    value_type var = 0;
    for (iterator_type it = first;it != last;++it) {
        value_type v = (*it - avg);
        var += v * v;
        ++n;
    }
    if (n > 0) {
        var /= n;
    }
    return var;
}

/**
 * A scheduler of online algorithms for training binary classifiers.
 *  This is a utility class to use online training algorithms from a data set
 *  transparently as a batch training algorithm.
 *
 *  @param  data_tmpl       The type of a data set.
 *  @param  trainer_tmpl    The type of an online training algorithm.
 */
template <
    class data_tmpl,
    class trainer_tmpl
>
class online_scheduler_binary
{
public:
    /// The type representing a data set for training.
    typedef data_tmpl data_type;
    /// The type implementing a training algorithm.
    typedef trainer_tmpl trainer_type;

    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;
    /// The type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// The type representing an attribute in an instance.
    typedef typename instance_type::attribute_type attribute_type;
    /// The type representing a value.
    typedef typename instance_type::value_type value_type;
    /// The type implementing an error function.
    typedef typename trainer_type::error_type error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename trainer_type::model_type model_type;

protected:
    /// Trainer type.
    trainer_type m_trainer;
    /// The sample method.
    std::string m_sample;
    /// The maximum number of iterations.
    int m_max_iterations;
    /// The parameter for regularization.
    value_type m_c;
    /// The period to measure the improvement ratio of loss.
    int m_period;
    /// The epsilon for improvement ratio.
    value_type m_epsilon;

public:
    /**
     * Constructs the object.
     */
    online_scheduler_binary()
    {
        clear();
    }

    /**
     * Destructs the object.
     */
    virtual ~online_scheduler_binary()
    {
    }

    /**
     * Resets the internal states and parameters to default.
     */
    void clear()
    {
        m_trainer.clear();

        parameter_exchange& par = this->params();
        par.init("sample", &m_sample, "shuffle",
            "The method for sampling instances.");
        par.init("max_iterations", &m_max_iterations, 100,
            "The maximum number of iterations (epochs).");
        par.init("c", &m_c, 1,
            "Coefficient (C) for regularization.");
        par.init("period", &m_period, 10,
            "The period to measure the improvement ratio");
        par.init("epsilon", &m_epsilon, 1e-6,
            "The stopping criterion for the improvement ratio");
    }

    /**
     * Obtains the parameter interface.
     *  @return parameter_exchange& The parameter interface associated with
     *                              this algorithm.
     */
    parameter_exchange& params()
    {
        // Forward to the training algorithm.
        return m_trainer.params();
    }

    /**
     * Obtains a read-only access to the weight vector (model).
     *  @return const model_type&   The weight vector (model).
     */
    const model_type& model() const
    {
        return m_trainer.model();
    }

    /**
     * Trains a model on a data set.
     *  @param  data        The data set for training (and holdout evaluation).
     *  @param  os          The output stream for progress reports.
     *  @param  holdout     The group number for holdout evaluation. Specify
     *                      a negative value if a holdout evaluation is
     *                      unnecessary.
     */
    void train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1
        )
    {
        // Translate the C parameter to an algorithm-specific parameter.
        parameter_exchange& par = this->params();
        par.set("n", (double)data.size(), false);

        // Reserve the weight vector.
        m_trainer.set_num_features(data.num_features());

        // Show the algorithm name and parameters.
        m_trainer.copyright(os);
        m_trainer.params().show(os);
        os << std::endl;

        // Initialize the training algorithm.
        m_trainer.start();

        std::vector<value_type> pf(m_period);

        // Loop for iterations.
        for (int k = 1;k <= m_max_iterations;++k) {
            value_type loss = 0;
            value_type avg = 0, var = 0;
            clock_t clk = std::clock();

            // Send instances to the algorithm.
            if (m_sample == "random") {
                // Choose N instances at random.
                for (size_t i = 0;i < data.size();++i) {
                    const_iterator it = random_sample(data.begin(), data.end());
                    if (it->get_group() != holdout) {
                        m_trainer.update(it);
                    }
                }
            } else if (m_sample == "cycle") {
                // Do not change the ordering of instances.
                for (const_iterator it = data.begin();it != data.end();++it) {
                    if (it->get_group() != holdout) {
                        m_trainer.update(it);
                    }
                }
            } else if (m_sample == "shuffle") {
                // Shuffle N instances first.
                std::vector<const_iterator> perm(data.size());
                shuffle_permutation(perm, data.begin(), data.end());
                for (size_t i = 0;i < perm.size();++i) {
                    const_iterator it = perm[i];
                    if (it->get_group() != holdout) {
                        m_trainer.update(it);
                    }
                }
            } else {
                throw invalid_parameter("Unknown sampling method for instances");
            }

            m_trainer.discontinue();
            loss = m_trainer.loss();

            pf[(k-1) % m_period] = loss;

            if (m_period < k) {
                avg = std::accumulate(pf.begin(), pf.end(), 0.);
                avg /= pf.size();
                var = compute_variance(pf.begin(), pf.end(), avg);
            }

            // Report the progress.
            os << "***** Iteration #" << k << " *****" << std::endl;
            m_trainer.report(os);
            if (m_period+1 < k) {
                os << "Moving average: " << avg << std::endl;
                os << "Variance: " << var << std::endl;
                os << "Variance / loss" << (var / loss) << std::endl;
            }
            os << "Seconds required for this iteration: " <<
                (std::clock() - clk) / (double)CLOCKS_PER_SEC << std::endl;

            // Holdout evaluation if necessary.
            if (0 <= holdout) {
                error_type cla(m_trainer.model());
                holdout_evaluation_binary(
                    os,
                    data.begin(),
                    data.end(),
                    cla,
                    holdout
                    );
            }

            // Flush the output stream.
            os << std::endl;
            os.flush();

            // Terminate if the stopping criterion is satisfied.
            if (avg < m_epsilon) {
                os << "Terminated with the stopping criterion" << std::endl;
                os << std::endl;
                os.flush();
                break;
            }
        }

        // Finalize the training procedure.
        m_trainer.finish();
    }
};



/**
 * A scheduler of online algorithms for training multi/candidate classifiers.
 *  This is a utility class to use online training algorithms from a data set
 *  transparently as a batch training algorithm.
 *
 *  @param  data_tmpl       The type of a data set.
 *  @param  trainer_tmpl    The type of an online training algorithm.
 */
template <
    class data_tmpl,
    class trainer_tmpl
>
class online_scheduler_multi
{
public:
    /// The type representing a data set for training.
    typedef data_tmpl data_type;
    /// The type implementing a training algorithm.
    typedef trainer_tmpl trainer_type;

    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;
    /// The type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// The type representing an attribute in an instance.
    typedef typename data_type::attribute_type attribute_type;
    /// The type representing a value.
    typedef typename instance_type::value_type value_type;
    /// The type implementing an error function.
    typedef typename trainer_type::error_type error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename trainer_type::model_type model_type;

protected:
    /// Trainer type.
    trainer_type m_trainer;
    /// The sample method.
    std::string m_sample;
    /// The maximum number of iterations.
    int m_max_iterations;
    /// The parameter for regularization.
    value_type m_c;
    /// The period to measure the improvement ratio of loss.
    int m_period;
    /// The epsilon for improvement ratio.
    value_type m_epsilon;

public:
    /**
     * Constructs the object.
     */
    online_scheduler_multi()
    {
        clear();
    }

    /**
     * Destructs the object.
     */
    virtual ~online_scheduler_multi()
    {
    }

    /**
     * Resets the internal states and parameters to default.
     */
    void clear()
    {
        m_trainer.clear();

        parameter_exchange& par = this->params();
        par.init("sample", &m_sample, "shuffle",
            "The method for sampling instances.");
        par.init("max_iterations", &m_max_iterations, 100,
            "The maximum number of iterations (epochs).");
        par.init("c", &m_c, 1,
            "Coefficient (C) for regularization.");
        par.init("period", &m_period, 10,
            "The period to measure the improvement ratio");
        par.init("epsilon", &m_epsilon, 1e-6,
            "The stopping criterion for the improvement ratio");
    }

    /**
     * Obtains the parameter interface.
     *  @return parameter_exchange& The parameter interface associated with
     *                              this algorithm.
     */
    parameter_exchange& params()
    {
        // Forward to the training algorithm.
        return m_trainer.params();
    }

    /**
     * Obtains a read-only access to the weight vector (model).
     *  @return const model_type&   The weight vector (model).
     */
    const model_type& model() const
    {
        return m_trainer.model();
    }

    /**
     * Trains a model on a data set.
     *  @param  data        The data set for training (and holdout evaluation).
     *  @param  os          The output stream for progress reports.
     *  @param  holdout     The group number for holdout evaluation. Specify
     *                      a negative value if a holdout evaluation is
     *                      unnecessary.
     */
    void train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1
        )
    {
        // Translate the C parameter to an algorithm-specific parameter.
        parameter_exchange& par = this->params();
        par.set("n", (double)data.size(), false);

        // Reserve the weight vector.
        m_trainer.set_num_features(data.num_features());

        // Show the algorithm name and parameters.
        m_trainer.copyright(os);
        m_trainer.params().show(os);
        os << std::endl;

        // Initialize the training algorithm.
        m_trainer.start();

        std::vector<value_type> pf(m_period);

        // Loop for iterations.
        for (int k = 1;k <= m_max_iterations;++k) {
            value_type loss = 0;
            value_type improvement = 0;
            clock_t clk = std::clock();

            // Send instances to the algorithm.
            if (m_sample == "random") {
                // Choose N instances at random.
                for (size_t i = 0;i < data.size();++i) {
                    const_iterator it = random_sample(data.begin(), data.end());
                    if (it->get_group() != holdout) {
                        m_trainer.update(
                            it, const_cast<data_type&>(data).feature_generator);
                    }
                }
            } else if (m_sample == "cycle") {
                // Do not change the ordering of instances.
                for (const_iterator it = data.begin();it != data.end();++it) {
                    if (it->get_group() != holdout) {
                        m_trainer.update(
                            it, const_cast<data_type&>(data).feature_generator);
                    }
                }
            } else if (m_sample == "shuffle") {
                // Shuffle N instances first.
                std::vector<const_iterator> perm(data.size());
                shuffle_permutation(perm, data.begin(), data.end());
                for (size_t i = 0;i < perm.size();++i) {
                    const_iterator it = perm[i];
                    if (it->get_group() != holdout) {
                        m_trainer.update(
                            it, const_cast<data_type&>(data).feature_generator);
                    }
                }
            } else {
                throw invalid_parameter("Unknown sampling method for instances");
            }

            m_trainer.discontinue();
            loss = m_trainer.loss();

            // Compute the improvement ratio.
            if (m_period < k) {
                improvement = (pf[(k-1) % m_period] - loss) / loss;
            } else {
                improvement = m_epsilon;
            }
            pf[(k-1) % m_period] = loss;

            // Report the progress.
            os << "***** Iteration #" << k << " *****" << std::endl;
            m_trainer.report(os);
            if (m_period < k) {
                os << "Improvement ratio: " << improvement << std::endl;
            }
            os << "Seconds required for this iteration: " <<
                (std::clock() - clk) / (double)CLOCKS_PER_SEC << std::endl;

            // Holdout evaluation if necessary.
            if (0 <= holdout) {
                error_type cla(m_trainer.model());
                holdout_evaluation_multi(
                    os,
                    data.begin(),
                    data.end(),
                    cla,
                    data.feature_generator,
                    holdout,
                    data.positive_labels.begin(),
                    data.positive_labels.end()
                    );
            }

            // Flush the output stream.
            os << std::endl;
            os.flush();

            // Terminate if the stopping criterion is satisfied.
            if (improvement < m_epsilon) {
                os << "Terminated with the stopping criterion" << std::endl;
                os << std::endl;
                os.flush();
                break;
            }
        }

        // Finalize the training procedure.
        m_trainer.finish();
    }
};

};

};

#endif/*__CLASSIAS_TRAIN_ONLINE_SCHEDULER_H__*/
