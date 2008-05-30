#ifndef __CLASSIAS_FEATURE_H__
#define __CLASSIAS_FEATURE_H__

#include <algorithm>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include "base.h"

namespace classias {

template <class features_type, class data_iterator_type>
void generate_sparse_features(
    features_type& features, data_iterator_type begin, data_iterator_type end)
{
    typedef typename data_iterator_type::value_type instance_type;
    typedef typename instance_type::attributes_type attributes_type;
    typedef typename attributes_type::const_iterator instance_iterator_type;

    for (data_iterator_type it = begin;it != end;++it) {
        for (instance_iterator_type ita = it->begin();ita != it->end();++ita) {
            features.associate(ita->first, it->get_label());
        }
    }
}

template <
    class ranking_instance_type,
    class features_type,
    class label_quark_type,
    class classification_instance_type
>
inline void
cinstance_to_rinstance(
    ranking_instance_type& ri,
    features_type& features,
    label_quark_type& labels,
    classification_instance_type& ci
    )
{
    typedef typename label_quark_type::value_type label_type;
    typedef typename classification_instance_type::attributes_type attributes_type;
    typedef typename attributes_type::const_iterator attributes_iterator;
    typedef typename ranking_instance_type::candidate_type ranking_candidate_type;

    // Loop over the possible labels.
    for (label_type l = 0;l < labels.size();++l) {
        ranking_candidate_type& cand = ri.new_element();

        // Translate the label for the candidate.
        bool is_true = (ci.get_label() == l);
        bool is_positive = is_positive_label(labels.to_item(ci.get_label()));
        cand.set_label(is_true, is_positive);

        // Append the features for the candidate.
        for (attributes_iterator it = ci.begin();it != ci.end();++it) {
            if (features.exists(it->first, l)) {
                cand.append(features.to_value(it->first, l), it->second);
            }
        }
    }
}

template <
    class ranking_instance_type,
    class features_type,
    class label_quark_type,
    class selection_instance_type
>
inline void
sinstance_to_rinstance(
    ranking_instance_type& ri,
    features_type& features,
    label_quark_type& labels,
    selection_instance_type& si
    )
{
    typedef typename selection_instance_type::labels_type labels_type;
    typedef typename labels_type::const_iterator labels_iterator;
    typedef typename selection_instance_type::attributes_type attributes_type;
    typedef typename attributes_type::const_iterator attributes_iterator;
    typedef typename ranking_instance_type::candidate_type ranking_candidate_type;

    for (labels_iterator itl = si.labels.begin();itl != si.labels.end();++itl) {
        ranking_candidate_type& cand = ri.new_element();

        // Translate the label for the candidate.
        bool is_true = (si.get_label() == itl->get_label());
        bool is_positive = is_positive_label(labels.to_item(itl->get_label()));
        cand.set_label(is_true, is_positive);

        // Append the features for the candidate.
        for (attributes_iterator it = si.begin();it != si.end();++it) {
            if (features.exists(it->first, itl->get_label())) {
                cand.append(features.to_value(it->first, itl->get_label()), it->second);
            }
        }
    }
}

template <
    class ranking_data_type,
    class features_type,
    class attribute_quark_type,
    class label_quark_type,
    class data_iterator_type
>
void classification_to_ranking(
    ranking_data_type& rd,
    features_type& features,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    data_iterator_type begin,
    data_iterator_type end
    )
{
    typedef typename data_iterator_type::value_type instance_type;
    typedef typename ranking_data_type::value_type ranking_instance_type;

    // Loop over the instances.
    for (data_iterator_type it = begin;it != end;++it) {
        // Construct the new instance within the vector.
        rd.resize(rd.size()+1);
        ranking_instance_type& ri = rd.back();

        // Copy the group identifier.
        ri.set_group(it->get_group());

        cinstance_to_rinstance(ri, features, labels, *it);
    }
}

template <
    class ranking_data_type,
    class features_type,
    class attribute_quark_type,
    class label_quark_type,
    class data_iterator_type
>
void selection_to_ranking(
    ranking_data_type& rd,
    features_type& features,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    data_iterator_type begin,
    data_iterator_type end
    )
{
    typedef typename data_iterator_type::value_type instance_type;
    typedef typename ranking_data_type::value_type ranking_instance_type;

    // Loop over the instances.
    for (data_iterator_type it = begin;it != end;++it) {
        // Construct the new instance within the vector.
        rd.resize(rd.size()+1);
        ranking_instance_type& ri = rd.back();

        // Copy the group identifier.
        ri.set_group(it->get_group());

        sinstance_to_rinstance(ri, features, labels, *it);
    }
}

template <
    class feature_type,
    class value_type,
    class attribute_quark_type,
    class label_quark_type
>
void
output_model(
    std::ostream& os,
    feature_type& features,
    const value_type* weights,
    attribute_quark_type& attrs,
    label_quark_type& labels
    )
{
    typedef typename feature_type::value_type featureid_type;

    for (featureid_type i = 0;i < features.size();++i) {
        int a, l;
        value_type w = weights[i];
        features.to_item(i, a, l);

        if (w != 0.) {
            os <<
                w << '\t' <<
                attrs.to_item(a) << '\t' <<
                labels.to_item(l) << std::endl;
        }
    }
}

};

#endif/*__CLASSIAS_FEATURE_H__*/
