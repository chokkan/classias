#ifndef __ATTRIBUTE_H__
#define __ATTRIBUTE_H__

#include <fstream>
#include <ctime>
#include <sstream>
#include <string>
#include <exception>

class invalid_data : public std::exception
{
protected:
    std::string message;

public:
    invalid_data(const char *const& msg, int lines)
    {
        std::stringstream ss;
        ss << "in lines " << lines << ", " << msg;
        message = ss.str();
    }

    invalid_data(const invalid_data& rho)
    {
        message = rho.message;
    }

    invalid_data& operator=(const invalid_data& rho)
    {
        message = rho.message;
    }

    virtual ~invalid_data() throw()
    {
    }

    virtual const char *what() const throw()
    {
        return message.c_str();
    }
};

class invalid_algorithm : public std::domain_error
{
public:
    explicit invalid_algorithm(const std::string& msg)
        : std::domain_error(msg)
    {
    }
};

class stopwatch
{
protected:
    clock_t begin;
    clock_t end;

public:
    stopwatch()
    {
        start();
    }

    void start()
    {
        begin = end = std::clock();
    }

    double stop()
    {
        end = std::clock();
        return get();
    }

    double get() const
    {
        return (end - begin) / (double)CLOCKS_PER_SEC;
    }
};

static void
get_name_value(
    const std::string& str, std::string& name, double& value)
{
    size_t col = str.rfind(':');
    if (col == str.npos) {
        name = str;
        value = 1.;
    } else {
        value = std::atof(str.c_str() + col + 1);
        name = str.substr(0, col);
    }
}

template <class data_type>
static void
read_data(
    data_type& data,
    const option& opt
    )
{
    std::ostream& os = std::cout;
    std::ostream& es = std::cerr;

    // Read files for training data.
    if (opt.files.empty()) {
        // Read the data from STDIN.
        os << "STDIN" << std::endl;
        read_stream(std::cin, data, opt, 0);
    } else {
        // Read the data from files.
        for (int i = 0;i < (int)opt.files.size();++i) {
            std::ifstream ifs(opt.files[i].c_str());
            if (!ifs.fail()) {
                os << "File (" << i+1 << "/" << opt.files.size() << ") : " << opt.files[i] << std::endl;
                read_stream(ifs, data, opt, i);
            }
            ifs.close();
        }
    }
}

template <class data_type>
static int
split_data(
    data_type& data,
    const option& opt
    )
{
    int i = 0;
    typename data_type::iterator it;
    for (it = data.begin();it != data.end();++it, ++i) {
        it->set_group(i % opt.split);
    }
    return opt.split;
}

template <class data_type>
static void
balance_instances(
    data_type& data
    )
{
    double 
    int i = 0;
    typename data_type::iterator it;
    for (it = data.begin();it != data.end();++it, ++i) {
        it->set_group(i % opt.split);
    }
    return opt.split;
}

template <class data_type>
static int
read_dataset(
    data_type& data,
    const option& opt
    )
{
    // Read the training data.
    read_data(data, opt);

    // Split the training data if necessary.
    if (0 < opt.split) {
        split_data(data, opt);
        return opt.split;
    } else {
        return (int)opt.files.size();
    }
}

template <class data_type>
static std::string
set_positive_labels(
    data_type& data,
    const option& opt
    )
{
    std::string negative_labels;
    typedef typename data_type::instance_type instance_type;
    typedef typename instance_type::label_type label_type;

    for (label_type l = 0;l < data.labels.size();++l) {
        const std::string& label = data.labels.to_item(l);
        if (opt.negatives.find(label) == opt.negatives.end()) {
            data.positive_labels.push_back(l);
        } else {
            if (!negative_labels.empty()) {
                negative_labels += ' ';
            }
            negative_labels += label;
        }
    }

    return negative_labels;
}

template <class char_type, class traits_type>
inline std::basic_ostream<char_type, traits_type>&
timestamp(std::basic_ostream<char_type, traits_type>& os)
{
	time_t ts;
	time(&ts);

   	char str[80];
    std::strftime(
        str, sizeof(str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&ts));

    os << str;
    return (os);
}

#endif/*__ATTRIBUTE_H__*/
