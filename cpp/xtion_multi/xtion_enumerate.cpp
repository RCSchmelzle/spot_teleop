#include <OpenNI.h>
#include <iostream>

int main() {
    // 1. Initialize OpenNI
    openni::Status rc = openni::OpenNI::initialize();
    if (rc != openni::STATUS_OK) {
        std::cerr << "Initialize failed:\n"
                  << openni::OpenNI::getExtendedError() << std::endl;
        return 1;
    }

    // 2. Enumerate all connected devices (returns void in this OpenNI2 version)
    openni::Array<openni::DeviceInfo> deviceList;
    openni::OpenNI::enumerateDevices(&deviceList);

    std::cout << "Found " << deviceList.getSize() << " device(s):" << std::endl;

    // 3. Print info for each device
    for (int i = 0; i < deviceList.getSize(); ++i) {
        const openni::DeviceInfo& info = deviceList[i];

        std::cout << "Device " << i << ":\n";
        std::cout << "  URI:           " << info.getUri() << "\n";
        std::cout << "  Vendor:        " << info.getVendor() << "\n";
        std::cout << "  Name:          " << info.getName() << "\n";
        std::cout << "  USB Vendor ID: " << info.getUsbVendorId() << "\n";
        std::cout << "  USB Product ID:" << info.getUsbProductId() << "\n";
    }

    openni::OpenNI::shutdown();
    return 0;
}
