#ifndef __CLASSIAS_FEATURE_H__
#define __CLASSIAS_FEATURE_H__

#include <iostream>
#include <string>

#include "base.h"
#include "quark.h"

namespace classias {

template <class features_type, class data_iterator_type>
void generate_sparse_features(
    features_type& features, data_iterator_type begin, data_iterator_type end)
{
    typedef typename data_iterator_type::value_type instance_type;
    typedef typename instance_type::attributes_type attributes_type;
    typedef typename attributes_type::const_iterator instance_iterator_type;

    for (data_iterator_type it = begin;it != end;++it) {
        for (instance_iterator_type ita = it->attributes.begin();ita != it->attributes.end();++ita) {
            features.associate(ita->first, it->label);
        }
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
