#include <OpenNI.h>
#include <iostream>
#include <fstream>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: xtion_snapshot <device_uri> <output_ppm_path>" << std::endl;
        return 1;
    }

    const char* uri = argv[1];
    const char* out_path = argv[2];

    // Initialize OpenNI
    openni::Status rc = openni::OpenNI::initialize();
    if (rc != openni::STATUS_OK) {
        std::cerr << "Initialize failed:\n"
                  << openni::OpenNI::getExtendedError() << std::endl;
        return 1;
    }

    // Open device
    openni::Device device;
    rc = device.open(uri);
    if (rc != openni::STATUS_OK) {
        std::cerr << "Failed to open device " << uri << ":\n"
                  << openni::OpenNI::getExtendedError() << std::endl;
        openni::OpenNI::shutdown();
        return 1;
    }

    // Create color stream
    openni::VideoStream color;
    rc = color.create(device, openni::SENSOR_COLOR);
    if (rc != openni::STATUS_OK) {
        std::cerr << "Failed to create color stream:\n"
                  << openni::OpenNI::getExtendedError() << std::endl;
        device.close();
        openni::OpenNI::shutdown();
        return 1;
    }

    rc = color.start();
    if (rc != openni::STATUS_OK) {
        std::cerr << "Failed to start color stream:\n"
                  << openni::OpenNI::getExtendedError() << std::endl;
        color.destroy();
        device.close();
        openni::OpenNI::shutdown();
        return 1;
    }

    // Read a few frames to let exposure settle
    openni::VideoFrameRef frame;
    for (int i = 0; i < 5; ++i) {
        rc = color.readFrame(&frame);
        if (rc != openni::STATUS_OK) {
            continue;
        }
        if (frame.isValid()) break;
    }

    if (!frame.isValid()) {
        std::cerr << "No valid color frame received." << std::endl;
        color.stop();
        color.destroy();
        device.close();
        openni::OpenNI::shutdown();
        return 1;
    }

    if (frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888) {
        std::cerr << "Unexpected pixel format, expected RGB888" << std::endl;
        color.stop();
        color.destroy();
        device.close();
        openni::OpenNI::shutdown();
        return 1;
    }

    int width = frame.getWidth();
    int height = frame.getHeight();
    const std::uint8_t* data = static_cast<const std::uint8_t*>(frame.getData());
    int expected_size = width * height * 3;

    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << out_path << std::endl;
        color.stop();
        color.destroy();
        device.close();
        openni::OpenNI::shutdown();
        return 1;
    }

    // Write PPM header
    ofs << "P6\n" << width << " " << height << "\n255\n";
    // Write pixel data
    ofs.write(reinterpret_cast<const char*>(data), expected_size);
    ofs.close();

    // Cleanup
    color.stop();
    color.destroy();
    device.close();
    openni::OpenNI::shutdown();

    return 0;
}
