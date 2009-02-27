#ifndef __CLASSIAS_CLASSIAS_H__
#define __CLASSIAS_CLASSIAS_H__

#include "base.h"
#include "traits.h"
#include "instance.h"
#include "data.h"

namespace classias
{

typedef feature_data_traits_base<int, int, int> feature_data_traits;
typedef dense_data_traits_base<int, int, int> dense_data_traits;
typedef sparse_data_traits_base<int, int, int> sparse_data_traits;

typedef sparse_vector_base<int, double> sparse_attributes;

typedef binary_instance_base<sparse_attributes, feature_data_traits> binstance;
typedef binary_data_base<binstance, quark> bdata;

typedef multi_candidate_base<sparse_attributes, int> mcandidate;
typedef multi_instance_base<mcandidate, feature_data_traits> minstance;
typedef multi_data_base<minstance, quark, quark> mdata;

typedef attribute_instance_base<sparse_attributes, int, sparse_data_traits> ainstance;
typedef attribute_data_base<ainstance, quark, quark> adata;

};

#endif/*__CLASSIAS_CLASSIAS_H__*/
