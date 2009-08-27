/*
 *      Averaged perceptron for binary classification.
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

#ifndef __CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_BINARY_H__
#define __CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_BINARY_H__

#include <iostream>

#include <classias/types.h>
#include <classias/classify/linear/binary.h>

namespace classias
{

namespace train
{

template <
    class error_tmpl,
    class model_tmpl
>
class averaged_perceptron_binary_base
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef model_tmpl model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// This class.
    typedef averaged_perceptron_binary_base<error_tmpl, model_tmpl> this_class;

protected:
    /// The array of feature weights.
    model_type m_w;
    model_type m_ws;
    bool m_averaged;

    int m_t;
    int m_c;

    /// Parameter interface.
    parameter_exchange m_params;

public:
    averaged_perceptron_binary_base()
    {
        clear();
    }

    virtual ~averaged_perceptron_binary_base()
    {
    }

    void clear()
    {
        // Clear the weight vector.
        m_w.clear();
        m_ws.clear();
        this->initialize_weights();
    }

public:
    void set_num_features(size_t size)
    {
        m_w.resize(size);
        m_ws.resize(size);
        this->initialize_weights();
    }

public:
    void start()
    {
        this->initialize_weights();
        m_t = 0;
        m_c = 1;
    }

    void finish()
    {
        average_weights();
    }

public:
    template <class iterator_type>
    value_type update(iterator_type it)
    {
        value_type loss = 0;

        error_type cls(m_w);
        cls.inner_product(it->begin(), it->end());
        if (static_cast<bool>(cls) != it->get_label()) {
            int y = static_cast<int>(it->get_label()) * 2 - 1;
            value_type delta = y * it->get_weight();
            update_weights(m_w, it->begin(), it->end(), delta);
            update_weights(m_ws, it->begin(), it->end(), m_c * delta);
            loss = 1;
        } else {
            loss = 0;
        }

        ++m_c;
        ++m_t;

        return loss;
    }

    template <class iterator_type>
    inline value_type update(iterator_type first, iterator_type last)
    {
        value_type loss = 0;
        for (iterator_type it = first;it != last;++it) {
            loss += this->update(it);
        }
        return loss;
    }

public:
    void copyright(std::ostream& os)
    {
        os << "Averaged perceptron (binary)" << std::endl;
    }

    void report(std::ostream& os)
    {
    }

protected:
    template <class iterator_type>
    inline void update_weights(
        model_type& w,
        iterator_type first,
        iterator_type last,
        value_type delta
        )
    {
        for (iterator_type it = first;it != last;++it) {
            w[it->first] += (delta * it->second);
        }
    }

    void initialize_weights()
    {
        for (size_t i = 0;i < m_w.size();++i) {
            m_w[i] = 0.;
            m_ws[i] = 0.;
        }
        m_c = 1;
    }

    void average_weights()
    {
        if (!m_averaged) {
            for (size_t i = 0;i < m_w.size();++i) {
                m_w[i] -= m_ws[i] / m_c;
            }
            for (size_t i = 0;i < m_w.size();++i) {
                m_ws[i] = m_w[i];
            }
            m_averaged = true;
        }
    }

public:
    parameter_exchange& params()
    {
        return m_params;
    }

public:
    model_type& model()
    {
        this->average_weights();
        return m_w;
    }

    const model_type& model() const
    {
        // Force to remove the const modifier for rescaling.
        return const_cast<this_class*>(this)->model();
    }
};

typedef averaged_perceptron_binary_base<
    classify::linear_binary<int, double, weight_vector>,
    weight_vector
    > averaged_perceptron_binary;

};

};

#endif/*__CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_BINARY_H__*/
