#ifndef __CLASSIAS_LBFGS_BINARY_H__
#define __CLASSIAS_LBFGS_BINARY_H__

#include <float.h>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include "base.h"
#include "../../evaluation.h"

namespace classias
{

template <
    class key_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_binary_classifier
{
public:
    typedef key_tmpl key_type;
    typedef value_tmpl value_type;
    typedef model_tmpl model_type;

protected:
    model_type& m_model;
    value_type m_score;

public:
    linear_binary_classifier(model_type& model)
        : m_model(model)
    {
        clear();
    }

    virtual ~linear_binary_classifier()
    {
    }

    inline operator bool() const
    {
        return (0. < m_score);
    }

    inline void operator()(const key_type& key, const value_type& value)
    {
        m_score += m_model[key] * value;        
    }

    template <class iterator_type>
    inline void inner_product(iterator_type first, iterator_type last)
    {
        m_score = 0.;
        for (iterator_type it = first;it != last;++it) {
            this->operator()(it->first, it->second);
        }
    }

    inline void clear()
    {
        m_score = 0.;
    }

    inline value_type score() const
    {
        return m_score;
    }

    inline value_type logistic_prob() const
    {
        return (-100. < m_score ? (1. / (1. + std::exp(-m_score))) : 0.);
    }

    inline value_type logistic_error(bool b) const
    {
        double p = 0.;
        if (m_score < -100.) {
            p = 0.;
        } else if (100. < m_score) {
            p = 1.;
        } else {
            p = 1. / (1. + std::exp(-m_score));
        }
        return (static_cast<double>(b) - p);
    }

    inline value_type logistic_error(bool b, value_type& logp) const
    {
        double p = 0.;
        if (m_score < -100.) {
            p = 0.;
            logp = static_cast<double>(b) * m_score;
        } else if (100. < m_score) {
            p = 1.;
            logp = (static_cast<double>(b) - 1.) * m_score;
        } else {
            p = 1. / (1. + std::exp(-m_score));
            logp = b ? std::log(p) : std::log(1.-p);
        }
        return (static_cast<double>(b) - p);
    }
};


/**
 * Training a logistic regression model.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_logress : public lbfgs_base<value_tmpl>
{
public:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A synonym of the base class.
    typedef lbfgs_base<value_tmpl> base_class;
    /// A synonym of this class.
    typedef trainer_logress<data_type, value_type> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

    /// A type representing a vector of features.
    typedef typename instance_type::features_type features_type;
    /// A type representing a feature identifier.
    typedef typename features_type::identifier_type feature_identifier_type;
    /// A classifier type.
    typedef linear_binary_classifier<feature_identifier_type, value_type, value_type const*> classifier_type;

protected:
    /// A data set for training.
    const data_type* m_data;

public:
    trainer_logress()
    {
    }

    virtual ~trainer_logress()
    {
    }

    void clear()
    {
        m_data = NULL;
        base_class::clear();
    }

    virtual value_type loss_and_gradient(
        const value_type *x,
        value_type *g,
        const int n
        )
    {
        typename data_type::const_iterator iti;
        typename instance_type::const_iterator it;
        value_type loss = 0;
        classifier_type cls(x);

        // For each instance in the data.
        for (iti = m_data->begin();iti != m_data->end();++iti) {
            // Exclude instances for holdout evaluation.
            if (iti->get_group() == m_holdout) {
                continue;
            }

            // Initialize the classifier.
            cls.clear();

            // Compute the score for the instance.
            cls.inner_product(iti->begin(), iti->end());

            // Compute the error.
            value_type logp = 0.;
            value_type d = cls.logistic_error(iti->get_truth(), logp);

            // Update the loss.
            loss -= iti->get_weight() * logp;

            // Update the gradients for the weights.
            d *= iti->get_weight();
            for (it = iti->begin();it != iti->end();++it) {
                g[it->first] -= d * it->second;
            }
        }

        return loss;
    }

    int train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1
        )
    {
        const size_t K = data.num_features();
        initialize_weights(K);
        
        os << "Training a logistic regression model with L-BFGS" << std::endl;
        m_params.show(os);
        os << std::endl;

        // Call the L-BFGS solver.
        m_data = &data;
        int ret = lbfgs_solve(
            (const int)K,
            os,
            holdout,
            data.get_user_feature_start()
            );

        // Report the result from the L-BFGS solver.
        lbfgs_output_status(os, ret);
        return ret;
    }

    void holdout_evaluation()
    {
        std::ostream& os = *m_os;
        const value_type *x = m_weights;
        int positive_labels[] = {1};
        confusion_matrix matrix(2);
        classifier_type cls(x);

        // For each instance in the data.
        for (const_iterator iti = m_data->begin();iti != m_data->end();++iti) {
            // Skip instances for training.
            if (iti->get_group() != m_holdout) {
                continue;
            }

            // Initialize the classifier.
            cls.clear();

            // Compute the score for the instance.
            cls.inner_product(iti->begin(), iti->end());

            int rl = static_cast<int>(iti->get_truth());
            int ml = static_cast<int>(static_cast<bool>(cls));

            // Store the results.
            matrix(rl, ml)++;
        }

        matrix.output_accuracy(os);
        matrix.output_micro(os, positive_labels, positive_labels+1);
    }
};

};

#endif/*__CLASSIAS_LBFGS_BINARY_H__*/