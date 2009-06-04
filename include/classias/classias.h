/*
 *		Classias library.
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

#ifndef __CLASSIAS_CLASSIAS_H__
#define __CLASSIAS_CLASSIAS_H__

#include "types.h"
#include "feature_generator.h"
#include "instance.h"
#include "data.h"

namespace classias
{

typedef dense_feature_generator_base<int_t, int_t, int_t> dense_feature_generator;
typedef sparse_feature_generator_base<int_t, int_t, int_t> sparse_feature_generator;

typedef sparse_vector_base<int_t, real_t> sparse_attributes;

typedef binary_instance_base<sparse_attributes> binstance;
typedef binary_data_base<binstance, quark> bdata;

typedef candidate_base<sparse_attributes, int_t> ccandidate;
typedef candidate_instance_base<ccandidate> cinstance;
typedef candidate_data_base<cinstance, quark, quark> cdata;

typedef multi_instance_base<sparse_attributes, int_t> minstance;
typedef multi_data_base<minstance, quark, quark, dense_feature_generator> mdata;

typedef multi_instance_base<sparse_attributes, int_t> ninstance;
typedef multi_data_base<ninstance, quark, quark, sparse_feature_generator> ndata;

};

#endif/*__CLASSIAS_CLASSIAS_H__*/
