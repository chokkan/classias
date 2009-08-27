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

#include <iostream>
#include <classias/evaluation.h>

namespace classias {

namespace train {

template <
    class data_tmpl,
    class trainer_tmpl
>
class online_scheduler_base
{
public:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    typedef trainer_tmpl trainer_type;

    typedef typename data_type::instance_type instance_type;

    typedef typename instance_type::attribute_type attribute_type;
    typedef typename instance_type::value_type value_type;

    typedef typename trainer_type::error_type error_type;
    typedef typename trainer_type::model_type model_type;

    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;

protected:
    /// Trainer type.
    trainer_type m_trainer;

    int m_max_iterations;
    value_type m_c;

public:
    online_scheduler_base()
    {
        clear();
    }

    virtual ~online_scheduler_base()
    {
    }

    void clear()
    {
        m_trainer.clear();

        parameter_exchange& par = this->params();
        par.init("max_iterations", &m_max_iterations, 100,
            "The maximum number of iterations (epochs).");
        par.init("c", &m_c, 1,
            "Coefficient (C) for L2-regularization.");
    }

    parameter_exchange& params()
    {
        // Forward to the training algorithm.
        return m_trainer.params();
    }

    const value_type* get_weights() const
    {
        return &m_trainer.model()[0];
    }

    int train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1
        )
    {
        // Translate the C parameter to an algorithm-specific parameter.
        parameter_exchange& par = this->params();
        if (par.get_stamp("lambda") <= par.get_stamp("c")) {
            par.set("lambda", 2. * m_c / data.size(), false);
        }

        // Reserve the weight vector.
        m_trainer.set_num_features(data.num_features());

        // Show the algorithm name and parameters.
        m_trainer.copyright(os);
        m_trainer.params().show(os);
        os << std::endl;

        // Initialize the training algorithm.
        m_trainer.start();

        // Loop for iterations.
        for (int k = 1;k <= m_max_iterations;++k) {
            value_type loss = 0;

            const_iterator it;
            for (it = data.begin();it != data.end();++it) {
                if (it->get_group() != holdout) {
                    loss += m_trainer.update(it);
                }
            }

            os << "***** Iteration #" << k << " *****" << std::endl;
            os << "Loss: " << loss << std::endl;
            m_trainer.report(os);

            if (holdout != -1) {
                this->holdout_evaluation(os, data, holdout);
            }

            os << std::endl;
            os.flush();
        }

        m_trainer.finish();

        return 0;
    }

    virtual void holdout_evaluation(std::ostream& os, const data_type& data, int holdout) = 0;
};

template <
    class data_tmpl,
    class trainer_tmpl
>
class online_scheduler_binary :
    public online_scheduler_base<data_tmpl, trainer_tmpl>
{
public:
    typedef data_tmpl data_type;
    typedef trainer_tmpl trainer_type;

    typedef typename trainer_type::error_type error_type;
    typedef typename trainer_type::model_type model_type;

public:
    online_scheduler_binary()
    {
    }

    virtual ~online_scheduler_binary()
    {
    }

    virtual void holdout_evaluation(std::ostream& os, const data_type& data, int holdout)
    {
        holdout_evaluation_binary<typename data_type::const_iterator, model_type, error_type>(
            os,
            data.begin(),
            data.end(),
            m_trainer.model(),
            holdout
            );
    }
};

};

};

#endif/*__CLASSIAS_TRAIN_ONLINE_SCHEDULER_H__*/
