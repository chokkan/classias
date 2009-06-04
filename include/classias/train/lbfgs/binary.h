#ifndef __CLASSIAS_LBFGS_BINARY_H__
#define __CLASSIAS_LBFGS_BINARY_H__

#include <float.h>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include "base.h"
#include <classias/classify/linear/binary.h>
#include "../../evaluation.h"

namespace classias
{

/**
 * Training a logistic regression model.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_lbfgs_binary : public lbfgs_base<value_tmpl>
{
public:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A type representing values for internal computations.
    typedef value_tmpl value_type;
    /// A synonym of the base class.
    typedef lbfgs_base<value_tmpl> base_class;
    /// A synonym of this class.
    typedef trainer_lbfgs_binary<data_tmpl, value_tmpl> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

    /// A type representing a vector of features.
    typedef typename instance_type::features_type features_type;
    /// A type representing a feature identifier.
    typedef typename features_type::identifier_type feature_identifier_type;
    /// A classifier type.
    typedef classify::linear_binary_logistic<feature_identifier_type, value_type, value_type const*> classifier_type;

protected:
    /// A data set for training.
    const data_type* m_data;

public:
    trainer_lbfgs_binary()
    {
    }

    virtual ~trainer_lbfgs_binary()
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

        // Initialize the gradients with zero.
        for (int i = 0;i < n;++i) {
            g[i] = 0.;
        }

        // For each instance in the data.
        for (iti = m_data->begin();iti != m_data->end();++iti) {
            // Exclude instances for holdout evaluation.
            if (iti->get_group() == this->m_holdout) {
                continue;
            }

            // Compute the score for the instance.
            cls.inner_product(iti->begin(), iti->end());

            // Compute the error.
            value_type nlogp = 0.;
            value_type err = cls.error(iti->get_truth(), nlogp);

            // Update the loss.
            loss += (iti->get_weight() * nlogp);

            // Update the gradients for the weights.
            err *= iti->get_weight();
            for (it = iti->begin();it != iti->end();++it) {
                g[it->first] += err * it->second;
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
        this->initialize_weights(K);
        
        os << "MAP estimation for a logistic regression model using L-BFGS" << std::endl;
        this->m_params.show(os);
        os << "lbfgs.regularization_start: " << data.get_user_feature_start() << std::endl;
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
        this->lbfgs_output_status(os, ret);
        return ret;
    }

    void holdout_evaluation()
    {
        std::ostream& os = *(this->m_os);
        const value_type *x = this->m_weights;
        int positive_labels[] = {1};
        accuracy acc;
        confusion_matrix matrix(2);
        classifier_type cls(x);

        // For each instance in the data.
        for (const_iterator iti = m_data->begin();iti != m_data->end();++iti) {
            // Skip instances for training.
            if (iti->get_group() != this->m_holdout) {
                continue;
            }

            // Compute the score for the instance.
            cls.inner_product(iti->begin(), iti->end());

            int rl = static_cast<int>(iti->get_truth());
            int ml = static_cast<int>(static_cast<bool>(cls));

            // Store the results.
            acc.set(rl == ml);
            matrix(rl, ml)++;
        }

        acc.output(os);
        matrix.output_micro(os, positive_labels, positive_labels+1);
    }
};

};

#endif/*__CLASSIAS_LBFGS_BINARY_H__*/

