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

    // Try to set the end index of the regularization.
    try {
        params.set("regularization.end", (int)data.get_user_feature_end());
    } catch (classias::unknown_parameter& e) {
        // Continue if the trainer does not support this parameter.
    }
}

template <
    class data_type,
    class value_type
>
static void
output_model(
    data_type& data,
    const value_type* weights,
    const option& opt
    )
{
    typedef typename data_type::features_quark_type features_quark_type;
    typedef typename features_quark_type::value_type features_type;
    const features_quark_type& features = data.features;

    // Open a model file for writing.
    std::ofstream os(opt.model.c_str());

    // Store the feature weights.
    for (features_type i = 0;i < features.size();++i) {
        value_type w = weights[i];
        if (w != 0.) {
            os << w << '\t' << features.to_item(i) << std::endl;
        }
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
    os << "Number of featuress: " << data.num_features() << std::endl;
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
        trainer.train(data, opt.os, -1);
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
