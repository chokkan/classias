#ifndef __TRAIN_H__
#define __TRAIN_H__

#include <vector>

#include <classias/feature.h>
#include <classias/maxent.h>

template <class data_type>
static int
train_maxent(
    data_type& data,
    int holdout,
    const option& opt
    )
{
    typedef typename classias::trainer_maxent<data_type> trainer_type;
    trainer_type trainer;

    // Set parameters.
    typename option::params_type::const_iterator itp;
    for (itp = opt.params.begin();itp != opt.params.end();++itp) {
        trainer.set(*itp);
    }

    trainer.train(data, opt.os, holdout);

    if (holdout == -1 && !opt.model.empty()) {
        output_model(data, trainer.get_weights(), opt);
    }

    return 0;
}


template <class data_type>
static int
train_a(option& opt)
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
    os << "Number of attributes: " << data.attributes.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            os << "Cross validation (" << (i + 1) << "/" << num_groups << ")" << std::endl;
            sw.start();
            train_maxent(data, i, opt);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        sw.start();
            train_maxent(data, -1, opt);
        sw.stop();
        os << "Seconds required: " << sw.get() << std::endl;
        os << std::endl;
    }

	// Report the finish time.
    os << "Finish time: " << timestamp << std::endl;
    os << std::endl;

    return 0;
}

template <class data_type>
static int
train_al(option& opt)
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
    os << "Number of attributes: " << data.attributes.size() << std::endl;
    os << "Number of labels: " << data.labels.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Generate features for the data.
    os << "Generating features for the data set." << std::endl;
    sw.start();
    classias::generate_sparse_features(data.features, data.begin(), data.end());
    sw.stop();
    os << "Number of features: " << data.features.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    data.num_labels = data.labels.size();

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            os << "Cross validation (" << (i + 1) << "/" << num_groups << ")" << std::endl;
            sw.start();
            train_maxent(data, i, opt);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        sw.start();
        train_maxent(data, -1, opt);
        sw.stop();
        os << "Seconds required: " << sw.get() << std::endl;
        os << std::endl;
    }

	// Report the finish time.
    os << "Finish time: " << timestamp << std::endl;
    os << std::endl;

    return 0;
}


#endif/*__TRAIN_H__*/
