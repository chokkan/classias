#ifndef __TRAIN_H__
#define __TRAIN_H__

#include <vector>

#include <classias/feature.h>
#include <classias/maxent.h>

template <
    class feature_type,
    class attribute_quark_type,
    class label_quark_type,
    class data_iterator_type
>
static int
train_maxent(
    feature_type& features,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    data_iterator_type begin,
    data_iterator_type end,
    int holdout,
    const option& opt
    )
{
    typedef typename classias::trainer_maxent<data_iterator_type> trainer_type;
    trainer_type trainer;

    // Set parameters.
    if (opt.algorithm == "L1") {
        trainer.set("sigma1", opt.sigma);
        trainer.set("sigma2", 0.);
    } else if (opt.algorithm == "L2") {
        trainer.set("sigma1", 0.);
        trainer.set("sigma2", opt.sigma);
    }

    trainer.train(begin, end, opt.os, holdout);

    if (holdout == -1 && !opt.model.empty()) {
        output_model(features, trainer.get_weights(), attrs, labels, opt);
    }

    return 0;
}


template <class instance_type>
static int
train_a(option& opt)
{
    typedef std::vector<instance_type> data_type;

    int no_features = 0;
    data_type data;
    int num_groups = 0;
    std::ostream& os = opt.os;
    classias::string_quark attrs;
    classias::string_quark labels;
    stopwatch sw;

	// Report the start time.
    os << "Start time: " << timestamp << std::endl;
    os << std::endl;

    // Read the source data.
    os << "Reading the data set" << std::endl;
    sw.start();
    num_groups = read_dataset(data, attrs, labels, opt);
    sw.stop();
    os << "Number of instances: " << data.size() << std::endl;
    os << "Number of groups: " << num_groups << std::endl;
    os << "Number of attributes: " << attrs.size() << std::endl;
    //os << "Number of labels: " << labels.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            os << "Cross validation (" << (i + 1) << "/" << num_groups << ")" << std::endl;
            sw.start();
            train_maxent(no_features, attrs, labels, data.begin(), data.end(), i, opt);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        sw.start();
        train_maxent(no_features, attrs, labels, data.begin(), data.end(), -1, opt);
        sw.stop();
        os << "Seconds required: " << sw.get() << std::endl;
        os << std::endl;
    }

	// Report the finish time.
    os << "Finish time: " << timestamp << std::endl;
    os << std::endl;

    return 0;
}

template <class instance_type>
static int
train_al(option& opt)
{
    typedef classias::quark2 feature_type;
    typedef std::vector<instance_type> raw_data_type;
    typedef std::vector<classias::rinstance> training_data_type;
    typedef training_data_type::const_iterator training_data_iterator_type;

    int num_groups = 0;
    std::ostream& os = opt.os;
    raw_data_type rawdata;
    feature_type features;
    classias::string_quark attrs;
    classias::string_quark labels;
    stopwatch sw;

	// Report the start time.
    os << "Start time: " << timestamp << std::endl;
    os << std::endl;

    // Read the source data.
    os << "Reading the data set" << std::endl;
    sw.start();
    num_groups = read_dataset(rawdata, attrs, labels, opt);
    sw.stop();
    os << "Number of instances: " << rawdata.size() << std::endl;
    os << "Number of groups: " << num_groups << std::endl;
    os << "Number of attributes: " << attrs.size() << std::endl;
    os << "Number of labels: " << labels.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Generate features for the data.
    os << "Generating features for the data set." << std::endl;
    sw.start();
    classias::generate_sparse_features(features, rawdata.begin(), rawdata.end());
    sw.stop();
    os << "Number of features: " << features.size() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Convert the data set to native training data.
    training_data_type trdata;
    os << "Converting the data set into native training data" << std::endl;
    sw.start();
    convert_to_ranking(trdata, features, attrs, labels, rawdata.begin(), rawdata.end());
    sw.stop();
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Free the source data.
    rawdata.clear();

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            os << "Cross validation (" << (i + 1) << "/" << num_groups << ")" << std::endl;
            sw.start();
            train_maxent(features, attrs, labels, trdata.begin(), trdata.end(), i, opt);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        sw.start();
        train_maxent(features, attrs, labels, trdata.begin(), trdata.end(), -1, opt);
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
