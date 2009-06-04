/*
 *		Utilities for training.
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

#ifndef __TRAIN_H__
#define __TRAIN_H__

#include <vector>
#include "util.h"

template <
    class trainer_type,
    class data_type>
static void
set_parameters(
    trainer_type& trainer,
    data_type& data,
    const option& opt
    )
{
    typename option::params_type::const_iterator itp;
    classias::parameter_exchange& params = trainer.params();
    for (itp = opt.params.begin();itp != opt.params.end();++itp) {
        std::string name, value;
        std::string::size_type pos = itp->find('=');
        if (pos != itp->npos) {
            name = std::string(*itp, 0, pos);
            value = itp->substr(pos+1);
        } else {
            name = *itp;
        }
        params.set(name, value);
    }
}

template <
    class data_type,
    class trainer_type
>
static int
train(option& opt)
{
    stopwatch sw;
    data_type data;
    int num_groups = 0;
    std::ostream& os = opt.os;

	// Report the start time.
    os << "Start time: " << timestamp << std::endl;
    os << std::endl;

    // Read the source data.
    os << "Reading the data set" << std::endl;
    sw.start();
    num_groups = read_dataset(data, opt);
    sw.stop();
    os << "Number of instances: " << data.size() << std::endl;
    os << "Number of groups: " << num_groups << std::endl;
    os << "Number of attributes: " << data.num_attributes() << std::endl;
    os << "Number of labels: " << data.num_labels() << std::endl;
    os << "Number of features: " << data.num_features() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            // Set training parameters.
            trainer_type trainer;
            set_parameters(trainer, data, opt);

            os << "Cross validation (" << (i + 1) << "/" << num_groups << ")" << std::endl;
            sw.start();
            trainer.train(data, opt.os, i);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        // Set training parameters.
        trainer_type trainer;
        set_parameters(trainer, data, opt);

        // Start training.
        sw.start();
        trainer.train(data, opt.os, (0 < opt.holdout ? (opt.holdout-1) : -1));
        sw.stop();
        os << "Seconds required: " << sw.get() << std::endl;
        os << std::endl;

        // Store the model.
        if (!opt.model.empty()) {
            output_model(data, trainer.get_weights(), opt);
        }
    }

	// Report the finish time.
    os << "Finish time: " << timestamp << std::endl;
    os << std::endl;

    return 0;
}

#endif/*__TRAIN_H__*/
