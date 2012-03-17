#ifndef OTL_EXCEPTION_43890928098790178928390128908493
#define OTL_EXCEPTION_43890928098790178928390128908493

class OTLException {
public:
    OTLException(std::string error_msg) {
        this->error_msg = error_msg;
    }
    void showError() {
        std::cerr << error_msg << std::endl;
    }
private:
    std::string error_msg;
};


#endif
