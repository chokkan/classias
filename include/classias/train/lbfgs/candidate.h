#ifndef __CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__
#define __CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__

#include <float.h>
#include <cmath>
#include <ctime>
#include <iostream>

#include "base.h"
#include "../../evaluation.h"

namespace classias
{

template <
    class key_tmpl,
    class label_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_candidate_classifier
{
public:
    typedef key_tmpl key_type;
    typedef key_tmpl label_type;
    typedef value_tmpl value_type;
    typedef model_tmpl model_type;

protected:
    typedef std::vector<value_type> scores_type;
    typedef std::vector<label_type> labels_type;

    model_type& m_model;
    scores_type m_scores;
    scores_type m_probs;
    labels_type m_labels;

    int         m_argmax;
    value_type  m_norm;

public:
    linear_candidate_classifier(model_type& model)
        : m_model(model)
    {
        clear();
    }

    virtual ~linear_candidate_classifier()
    {
    }

    inline void clear()
    {
        m_norm = 0.;
        for (int i = 0;i < this->size();++i) {
            m_scores[i] = 0.;
            m_probs[i] = 0.;
        }
    }

    inline void resize(int n)
    {
        m_scores.resize(n);
        m_probs.resize(n);
        m_labels.resize(n);
    }

    inline int size() const
    {
        return (int)m_scores.size();
    }

    inline int argmax() const
    {
        return m_argmax;
    }

    inline value_type score(int i)
    {
        return m_scores[i];
    }

    inline value_type prob(int i)
    {
        return m_probs[i];
    }

    inline const label_type& label(int i)
    {
        return m_labels[i];
    }

    inline void operator()(int i, const key_type& key, const value_type& value)
    {
        m_scores[i] += m_model[key] * value;
    }

    template <class iterator_type>
    inline void accumulate(int i, iterator_type first, iterator_type last, const label_type& label)
    {
        m_scores[i] = 0.;
        m_labels[i] = label;
        for (iterator_type it = first;it != last;++it) {
            this->operator()(i, it->first, it->second);
        }
    }

    template <class iterator_type>
    inline void add_to(value_type* v, iterator_type first, iterator_type last, value_type value)
    {
        for (iterator_type it = first;it != last;++it) {
            v[it->first] += value * it->second;
        }        
    }

    inline bool finalize(bool prob)
    {
        if (m_scores.size() == 0) {
            return false;
        }

        // Find the argmax index.
        m_argmax = 0;
        double vmax = m_scores[0];
        for (int i = 0;i < this->size();++i) {
            if (vmax < m_scores[i]) {
                m_argmax = i;
                vmax = m_scores[i];
            }
        }

        if (prob) {
            // Compute the exponents of scores.
            for (int i = 0;i < this->size();++i) {
                m_probs[i] = std::exp(m_scores[i]);
            }

            // Compute the partition factor, starting from the maximum value.
            m_norm = m_probs[m_argmax];
            for (int i = 0;i < this->size();++i) {
                if (i != m_argmax) {
                    m_norm += m_probs[i];
                }
            }

            // Normalize the probabilities.
            for (int i = 0;i < this->size();++i) {
                m_probs[i] /= m_norm;
            }
        }

        return true;
    }
};




/**
 * Training a log-linear model using the maximum entropy modeling.
 *  @param  data_tmpl           Training data class.
 *  @param  value_tmpl          The type for computation.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_lbfgs_candidate : public lbfgs_base<value_tmpl>
{
protected:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A synonym of the base class.
    typedef lbfgs_base<value_tmpl> base_class;
    /// A synonym of this class.
    typedef trainer_lbfgs_candidate<data_type, value_type> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    typedef typename data_type::attribute_type attribute_type;
    typedef typename instance_type::candidate_type candidate_type;
    /// A type representing a label.
    typedef typename data_type::label_type label_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

    typedef linear_candidate_classifier<attribute_type, label_type, value_type, value_type const*> classifier_type;


    /// An array [K] of observation expectations.
    value_type *m_oexps;
    /// An array [K] of model expectations.
    value_type *m_mexps;
    /// An array [M] of scores for candidate labels.
    value_type *m_scores;

    label_type m_num_labels;

    /// A data set for training.
    const data_type* m_data;

public:
    trainer_lbfgs_candidate()
    {
        m_oexps = NULL;
        m_mexps = NULL;
        m_scores = NULL;
        clear();
    }

    virtual ~trainer_lbfgs_candidate()
    {
        clear();
    }

    void clear()
    {
        delete[] m_mexps;
        delete[] m_oexps;
        delete[] m_scores;
        m_oexps = 0;
        m_mexps = 0;
        m_scores = 0;

        m_data = NULL;
        m_num_labels = 0;
        base_class::clear();
    }

    virtual value_type loss_and_gradient(
        const value_type *x,
        value_type *g,
        const int n
        )
    {
        int i;
        value_type loss = 0, norm = 0;
        const data_type& data = *m_data;
        classifier_type cls(x);

        cls.resize(m_num_labels);

        for (int i = 0;i < n;++i) {
            m_mexps[i] = 0.;
        }

        // For each instance in the data.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            int itrue = -1;
            const instance_type& inst = *iti;

            // Exclude instances for holdout evaluation.
            if (inst.get_group() == m_holdout) {
                continue;
            }

            // Compute score[i] for each candidate #i.
            for (i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                cls.accumulate(i, cand.begin(), cand.end(), cand.get_label());
                if (cand.get_truth()) {
                    itrue = i;
                }
            }

            cls.finalize(true);

            // Accumulate the model expectations of features.
            for (i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                cls.add_to(m_mexps, cand.begin(), cand.end(), cls.prob(i));
            }

            // Accumulate the loss for predicting the instance.
            loss -= std::log(cls.prob(itrue));
        }

        // Compute the gradients.
        for (int i = 0;i < n;++i) {
            g[i] = -(m_oexps[i] - m_mexps[i]);
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
        const size_t L = data.num_labels();

        // Initialize feature expectations and weights.
        initialize_weights(K);
        m_oexps = new double[K];
        m_mexps = new double[K];
        for (size_t k = 0;k < K;++k) {
            m_oexps[k] = 0.;
            m_mexps[k] = 0.;
        }

        // Report the training parameters.
        os << "Training a maximum entropy model" << std::endl;
        m_params.show(os);
        os << std::endl;

        // Compute observation expectations of the features.
        m_num_labels = L;
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            const instance_type& inst = *iti;

            // Skip instances for holdout evaluation.
            if (inst.get_group() == m_holdout) {
                continue;
            }

            // Compute the observation expectations.
            for (int i = 0;i < inst.size();++i) {
                const candidate_type& cand = inst[i];
                if (cand.get_truth()) {
                    typename candidate_type::const_iterator it;
                    for (it = cand.begin();it != cand.end();++it) {
                        m_oexps[it->first] += it->second;
                    }
                }
            }
        }

        // Initialze the variables used by callback functions.
        m_scores = new double[m_num_labels];

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
        const data_type& data = *m_data;
        accuracy acc;
        confusion_matrix matrix(data.labels.size());
        const value_type *x = m_weights;
        classifier_type cls(x);

        // Loop over instances.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            int itrue = -1;
            const instance_type& inst = *iti;

            // Exclude instances for holdout evaluation.
            if (inst.get_group() != m_holdout) {
                continue;
            }

            // Compute score[i] for each candidate #i.
            for (int i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                cls.accumulate(i, cand.begin(), cand.end(), cand.get_label());
                if (cand.get_truth()) {
                    itrue = i;
                }
            }

            cls.finalize(false);

            int idx_max = cls.argmax();
            acc.set(itrue == idx_max);
            matrix(cls.label(itrue), cls.label(idx_max))++;
        }

        // Report accuracy, precision, recall, and f1 score.
        acc.output(os);
        matrix.output_micro(os, data.positive_labels.begin(), data.positive_labels.end());
    }
};

};

#endif/*__CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__*/
